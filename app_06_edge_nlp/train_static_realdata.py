"""
App 06 edge NLP: train the static baseline on real SST-2 with SentencePiece.

This mirrors `train_asstf_realdata.py` but uses a standard TransformerEncoderLayer
FFN, giving a fair real-data comparison for the static baseline.
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

from asstf import count_parameters
from app_06_edge_nlp.train_asstf_realdata import load_data


class TinyBertStaticReal(nn.Module):
    """Standard Tiny-BERT-style encoder for real SST-2 + SentencePiece."""

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
        return self.pool(h[:, 0, :])


def run_epoch(model, opt, loader, train, device, loss_fn):
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


def main():
    batch_size = 32
    lr = 2e-4
    epochs = 200
    patience = 1000

    train_loader, test_loader, vocab_size = load_data(batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyBertStaticReal(vocab_size, hidden_size=128, num_layers=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    print("=" * 60)
    print("App 06 Edge NLP: Static baseline training on real SST-2 (SentencePiece)")
    print("=" * 60)
    print(f"Vocab size: {vocab_size}")
    print(f"Static params: {count_parameters(model):,}")

    best_val_acc = 0.0
    best_state = None
    no_improve = 0
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, opt, train_loader, True, device, loss_fn)
        val_loss, val_acc = run_epoch(model, None, test_loader, False, device, loss_fn)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[Static] Epoch {epoch+1:03d}: "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
            )
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"[Static] Early stopping at epoch {epoch+1} (best val acc {best_val_acc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    cp_dir = ROOT / "checkpoints" / "app_06_edge_nlp"
    cp_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cp_dir / "static_model_real.pt")

    _, test_acc = run_epoch(model, None, test_loader, False, device, loss_fn)

    result = {
        "app": "app_06_edge_nlp",
        "data": "real_sst2",
        "static_params": count_parameters(model),
        "static_test_acc": test_acc,
        "static_best_val_acc": best_val_acc,
    }
    out_dir = ROOT / "results" / "app_06_edge_nlp"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result_static_real.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\nResult:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
