"""
Inference-time structural adaptation via surprise minimisation.

Reference: Eq. (14)-(15) of the ASSTF white-paper.

    dθs/dt = -α (∂L/∂θs + β ∂S/∂θs)

    S = -log p(x | θs) + KL(q(θs) || p(θs))

In this reference implementation the negative log-likelihood is approximated by
a reconstruction loss (mean-squared error or cross-entropy) on the model's
internal representation or prediction, and the KL term is approximated by a
Gaussian prior regulariser on the structural parameters.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurpriseMinimizer:
    """
    Adapt structural parameters θs at inference time to minimise surprise on
    an incoming stream of inputs.

    The adaptation is unsupervised: only the input ``x`` is required.  It is
    particularly useful for personalisation, concept-drift compensation and
    online reinforcement learning.

    Parameters
    ----------
    model : nn.Module
        The model to adapt.  Only parameters classified as structural are
        updated.
    lr : float
        Adaptation learning rate (α in Eq. 14).
    beta : float
        Weight of the KL/prior regulariser (β in Eq. 14).
    reconstruction_loss : str
        One of ``"mse"`` or ``"ce"``.
    prior_std : float
        Standard deviation of the Gaussian prior over θs.
    max_steps : int
        Maximum number of adaptation steps per call.
    """

    STRUCTURAL_NAME_PATTERNS = (
        "structural_u",
        "structural_v",
        "structural_gate",
        "alpha",
        "zeta",
        "beta",
        "gate_scale",
        "gate_bias",
    )

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        beta: float = 0.01,
        reconstruction_loss: str = "mse",
        prior_std: float = 1.0,
        max_steps: int = 5,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.model = model
        self.lr = lr
        self.beta = beta
        self.prior_std = prior_std
        self.max_steps = max_steps
        self.reconstruction_loss = reconstruction_loss
        self.device = device or next(model.parameters()).device

        if reconstruction_loss == "mse":
            self.recon_loss = F.mse_loss
        elif reconstruction_loss == "ce":
            self.recon_loss = F.cross_entropy
        else:
            raise ValueError(f"Unknown reconstruction_loss: {reconstruction_loss}")

        self.struct_params = self._collect_structural_params()
        self.optimizer = torch.optim.SGD(self.struct_params, lr=self.lr)

    def _is_structural(self, name: str) -> bool:
        name_lower = name.lower()
        for suffix in self.STRUCTURAL_NAME_PATTERNS:
            if name_lower.endswith(suffix) or (suffix + "_") in name_lower or name_lower == suffix:
                return True
        return False

    def _collect_structural_params(self):
        params = []
        for name, param in self.model.named_parameters():
            if self._is_structural(name) and param.requires_grad:
                params.append(param)
        return params

    def _kl_prior(self) -> torch.Tensor:
        """
        Approximate KL regulariser KL(q(θs) || p(θs)) assuming a zero-mean
        Gaussian prior with std ``prior_std``.
        """
        kl = torch.tensor(0.0, device=self.device)
        for p in self.struct_params:
            kl = kl + (p ** 2).sum() / (2.0 * self.prior_std ** 2)
        return kl

    def adapt(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> float:
        """
        Perform online structural adaptation on a single batch ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Input batch.
        target : torch.Tensor | None
            Optional target for supervised adaptation.  If None, the model's
            own prediction is used as the reconstruction target (self-supervised).
        loss_fn : Callable | None
            Optional custom loss function.  If provided, overrides the default
            reconstruction loss.

        Returns
        -------
        float
            Final adaptation loss value.
        """
        # Inference-time adaptation should keep dropout / batch-norm in eval
        # mode.  Gradients are enabled below via the normal forward pass, not
        # by setting the global training flag.
        self.model.eval()

        # Freeze core parameters to ensure only θs are adapted.
        for name, param in self.model.named_parameters():
            if not self._is_structural(name):
                param.requires_grad = False
            else:
                param.requires_grad = True

        x = x.to(self.device)
        if target is not None:
            target = target.to(self.device)

        final_loss = 0.0
        for _ in range(self.max_steps):
            self.optimizer.zero_grad()
            out = self.model(x)

            if loss_fn is not None:
                # Custom loss is fully responsible for its target handling.
                loss = loss_fn(out, target)
            elif target is not None:
                loss = self.recon_loss(out, target)
            elif self.reconstruction_loss == "mse":
                # Self-supervised reconstruction: reproduce the input.
                loss = self.recon_loss(out, x)
            else:
                raise ValueError(
                    "Cross-entropy adaptation requires an explicit target tensor"
                )

            surprise = loss + self.beta * self._kl_prior()
            surprise.backward()
            torch.nn.utils.clip_grad_norm_(self.struct_params, max_norm=0.5)
            self.optimizer.step()
            final_loss = surprise.item()

        # Restore core parameter gradients.
        for param in self.model.parameters():
            param.requires_grad = True

        return final_loss

    def adapt_stream(
        self,
        stream_loader,
        steps_per_batch: int = 1,
    ) -> list:
        """
        Adapt sequentially on a stream of batches (e.g. sensor stream).

        Returns a list of per-batch adaptation losses.
        """
        losses = []
        for batch in stream_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, _ = batch
            else:
                x = batch
            for _ in range(steps_per_batch):
                loss = self.adapt(x)
            losses.append(loss)
        return losses
