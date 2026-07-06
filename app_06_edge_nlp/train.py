"""
App 06: Edge NLP Intent Classification - Training

This version trains on the real SST-2 dataset encoded with the custom
SentencePiece tokenizer.  Both the ASSTF model and the static baseline are
trained until their validation accuracy plateaus, and the ASSTF model uses
the BilevelTrainer to update core and structural parameters in alternating
phases (Algorithm 1).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from asstf import ASSTFLinear, BilevelTrainer, count_parameters
from shared import accuracy
from shared.early_stopping import EarlyStopping


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_sst2_sentencepiece(max_samples: int | None = None):
    """Load real SST-2 encoded by the custom SentencePiece tokenizer."""
    data = np.load(ROOT / "data" / "sst2_sentencepiece.npz")
    X = data["X"].astype(np.int64)
    y = data["y"].astype(np.int64)
    vocab_size = int(data["vocab_size"])
    if max_samples is not None and max_samples < len(X):
        X = X[:max_samples]
        y = y[:max_samples]
    return X, y, vocab_size


def load_data(batch_size: int = 32, test_split: float = 0.2, max_samples: int | None = None):
    X, y, vocab_size = load_sst2_sentencepiece(max_samples=max_samples)
    n_test = int(len(X) * test_split)
    idx = np.random.RandomState(42).permutation(len(X))
    train_idx, test_idx = idx[n_test:], idx[:n_test]

    train_ds = TensorDataset(
        torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx])
    )
    test_ds = TensorDataset(
        torch.from_numpy(X[test_idx]), torch.from_numpy(y[test_idx])
    )

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        vocab_size,
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TinyBertFFN(nn.Module):
    """A tiny BERT-style encoder block with standard FFN."""

    def __init__(self, vocab_size: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 2,
                dropout=0.1,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.pool = nn.Linear(hidden_size, 2)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        cls = h[:, 0, :]
        return self.pool(cls)


class TinyBertAllASSTF96(nn.Module):
    """Smaller ASSTF Tiny-BERT: hidden_size=96, num_layers=2, rank=4."""

    def __init__(self, vocab_size: int, hidden_size: int = 96, num_layers: int = 2, rank: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.norms1 = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads=4, dropout=0.1, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                ASSTFLinear(hidden_size, hidden_size, structural_rank=rank),
                nn.GELU(),
                ASSTFLinear(hidden_size, hidden_size, structural_rank=rank),
            )
            for _ in range(num_layers)
        ])
        self.pool = ASSTFLinear(hidden_size, 2, structural_rank=1)

    def forward(self, x):
        h = self.embed(x)
        for i in range(self.num_layers):
            attn_out, _ = self.attns[i](h, h, h, need_weights=False)
            h = self.norms1[i](h + attn_out)
            ffn_out = self.ffns[i](h)
            h = self.norms2[i](h + ffn_out)
        cls = h[:, 0, :]
        return self.pool(cls)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def run_epoch_static(model, opt, loader, train, device, loss_fn):
    model.train(train)
    total_loss = 0.0
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    epochs: int = 200,
    asstf_epochs: int | None = None,
    batch_size: int = 32,
    lr_core: float = 2e-4,
    lr_struct: float = 1e-4,
    static_patience: int = 1000,
    asstf_patience: int | None = None,
    max_samples: int | None = None,
    save_dir: Path | None = None,
):
    asstf_epochs = asstf_epochs or 500
    asstf_patience = asstf_patience or static_patience
    save_dir = save_dir or (ROOT / "checkpoints" / "app_06_edge_nlp")
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, vocab_size = load_data(batch_size, max_samples=max_samples)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    asstf_model = TinyBertAllASSTF96(vocab_size, hidden_size=96, num_layers=2, rank=4).to(device)
    static_model = TinyBertFFN(vocab_size, hidden_size=128, num_layers=2).to(device)

    print("=" * 60)
    print("App 06: Edge NLP Intent Classification")
    print("=" * 60)
    print(f"Data: real SST-2 + SentencePiece")
    print(f"Vocab size: {vocab_size}")
    print(f"ASSTF params:  {count_parameters(asstf_model):,}")
    print(f"Static params: {count_parameters(static_model):,}")

    loss_fn = nn.CrossEntropyLoss()

    # Static baseline: standard end-to-end optimizer.
    static_opt = torch.optim.AdamW(
        static_model.parameters(), lr=lr_core, weight_decay=1e-2
    )
    static_stopper = EarlyStopping(patience=static_patience, mode="max")

    for epoch in range(epochs):
        s_loss, s_acc = run_epoch_static(
            static_model, static_opt, train_loader, True, device, loss_fn
        )
        s_vloss, s_vacc = run_epoch_static(
            static_model, None, test_loader, False, device, loss_fn
        )
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[Static] Epoch {epoch+1:03d}: train_acc={s_acc:.4f} val_acc={s_vacc:.4f}"
            )
        if static_stopper(s_vacc):
            print(
                f"[Static] Early stopping at epoch {epoch+1} "
                f"(best val acc {static_stopper.best_value:.4f})"
            )
            break

    # ASSTF: alternating core / structural optimization.
    asstf_trainer = BilevelTrainer(
        asstf_model,
        lr_core=lr_core,
        lr_struct=lr_struct,
        weight_decay=1e-2,
        optimizer_cls=torch.optim.AdamW,
        device=device,
    )
    asstf_stopper = EarlyStopping(patience=asstf_patience, mode="max")

    for epoch in range(asstf_epochs):
        metrics = asstf_trainer.train_epoch(train_loader, loss_fn, alternate=True)
        eval_metrics = asstf_trainer.evaluate(
            test_loader, loss_fn, metric_fn=accuracy
        )
        a_vacc = eval_metrics["metric"]
        a_vloss = eval_metrics["loss"]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[ASSTF] Epoch {epoch+1:03d}: "
                f"core_loss={metrics['core_loss']:.4f} "
                f"struct_loss={metrics['struct_loss']:.4f} "
                f"val_acc={a_vacc:.4f}"
            )
        if asstf_stopper(a_vacc):
            print(
                f"[ASSTF] Early stopping at epoch {epoch+1} "
                f"(best val acc {asstf_stopper.best_value:.4f})"
            )
            break

    torch.save(asstf_model.state_dict(), save_dir / "asstf_model.pt")
    torch.save(static_model.state_dict(), save_dir / "static_model.pt")

    metadata = {
        "app": "app_06_edge_nlp",
        "data": "real_sst2_sentencepiece",
        "vocab_size": vocab_size,
        "epochs": epochs,
        "asstf_epochs": asstf_epochs,
        "static_patience": static_patience,
        "asstf_patience": asstf_patience,
        "asstf_params": count_parameters(asstf_model),
        "static_params": count_parameters(static_model),
        "asstf_val_acc": asstf_trainer.evaluate(
            test_loader, loss_fn, metric_fn=accuracy
        )["metric"],
        "static_val_acc": run_epoch_static(
            static_model, None, test_loader, False, device, loss_fn
        )[1],
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved models to", save_dir)
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    train()
