"""
Bilevel optimisation trainer for ASSTF networks.

Implements Algorithm 1 from the ASSTF white-paper:

    Phase 1 (core parameters):      θ  <- θ  - η1 ∇θ L
    Phase 2 (structural parameters): θs <- θs - η2 Dε L(θs)

The trainer automatically splits model parameters into core (θc) and
structural (θs) groups and updates them with separate optimisers.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _default_device() -> torch.device:
    """Pick the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BilevelTrainer:
    """
    Bilevel trainer for ASSTF models.

    Parameters
    ----------
    model : nn.Module
        A model containing ASSTF layers.  Structural parameters are detected
        by name (parameters whose names contain ``structural_`` or one of
        ``alpha``, ``zeta``, ``beta``, ``gate_scale``, ``gate_bias``).
    lr_core : float
        Learning rate for core parameters θc (Phase 1).
    lr_struct : float
        Learning rate for structural parameters θs (Phase 2).
    weight_decay : float
        L2 regularisation applied only to core parameters.
    optimizer_cls : type
        Optimiser class to use for both phases (default: torch.optim.Adam).
    device : torch.device | str | None
        Device to move batches to.  If None, use CUDA if available.
    """

    STRUCTURAL_NAME_PATTERNS = (
        "structural_",
        "alpha",
        "zeta",
        "beta",
        "gate_scale",
        "gate_bias",
    )

    def __init__(
        self,
        model: nn.Module,
        lr_core: float = 1e-3,
        lr_struct: float = 1e-4,
        weight_decay: float = 1e-5,
        optimizer_cls: type = torch.optim.Adam,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.model = model
        self.device = device or _default_device()
        self.model.to(self.device)

        core_params, struct_params = self._split_parameters(model)
        self.core_params = core_params
        self.struct_params = struct_params

        self.optimizer_core = optimizer_cls(core_params, lr=lr_core, weight_decay=weight_decay)
        self.optimizer_struct = optimizer_cls(struct_params, lr=lr_struct)

        self.history: Dict[str, list] = {"core_loss": [], "struct_loss": []}

    @classmethod
    def _is_structural(cls, name: str) -> bool:
        """Heuristic to decide whether a parameter belongs to θs."""
        name_lower = name.lower()
        # Exact suffix checks to avoid matching e.g. "alphabet_embedding".
        for suffix in cls.STRUCTURAL_NAME_PATTERNS:
            if name_lower.endswith(suffix) or (suffix + "_") in name_lower or name_lower == suffix:
                return True
        return False

    def _split_parameters(self, model: nn.Module) -> Tuple[Iterable, Iterable]:
        core, struct = [], []
        for name, param in model.named_parameters():
            if self._is_structural(name):
                struct.append(param)
            else:
                core.append(param)
        return core, struct

    def _phase_step(
        self,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        batch: Tuple[torch.Tensor, torch.Tensor],
        phase: str,
    ) -> float:
        """
        Run a single phase update.

        Parameters
        ----------
        phase : {"core", "struct"}
            Which parameter group to update.
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        if phase == "core":
            self.optimizer_core.zero_grad()
        else:
            self.optimizer_struct.zero_grad()

        logits = self.model(x)
        loss = loss_fn(logits, y)
        loss.backward()

        # Gradient clipping for stability.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        if phase == "core":
            self.optimizer_core.step()
        else:
            self.optimizer_struct.step()

        return loss.item()

    def train_epoch(
        self,
        dataloader: DataLoader,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        alternate: bool = True,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Parameters
        ----------
        alternate : bool
            If True, perform alternating Phase 1 / Phase 2 updates per batch
            (Algorithm 1).  If False, both groups are updated jointly every
            batch (useful for quick baseline experiments).
        """
        self.model.train()
        core_losses, struct_losses = [], []

        for batch in dataloader:
            if alternate:
                c_loss = self._phase_step(loss_fn, batch, phase="core")
                s_loss = self._phase_step(loss_fn, batch, phase="struct")
            else:
                # Joint update: still use two optimisers but reset them together.
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer_core.zero_grad()
                self.optimizer_struct.zero_grad()
                loss = loss_fn(self.model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer_core.step()
                self.optimizer_struct.step()
                c_loss = s_loss = loss.item()

            core_losses.append(c_loss)
            struct_losses.append(s_loss)

        avg_core = sum(core_losses) / len(core_losses)
        avg_struct = sum(struct_losses) / len(struct_losses)
        self.history["core_loss"].append(avg_core)
        self.history["struct_loss"].append(avg_struct)

        return {"core_loss": avg_core, "struct_loss": avg_struct}

    def evaluate(
        self,
        dataloader: DataLoader,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        metric_fn: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
    ) -> Dict[str, float]:
        """Evaluate the model on a validation/test set."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = loss_fn(logits, y)
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)

                all_preds.append(logits.detach().cpu())
                all_labels.append(y.detach().cpu())

        result = {"loss": total_loss / total_samples}

        if metric_fn is not None:
            # Custom metric provided by caller.
            result["metric"] = metric_fn(torch.cat(all_preds), torch.cat(all_labels))

        return result

    def state_dict(self) -> Dict:
        return {
            "model": self.model.state_dict(),
            "optimizer_core": self.optimizer_core.state_dict(),
            "optimizer_struct": self.optimizer_struct.state_dict(),
            "history": self.history,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.optimizer_core.load_state_dict(state_dict["optimizer_core"])
        self.optimizer_struct.load_state_dict(state_dict["optimizer_struct"])
        self.history = state_dict["history"]
