"""
App 06 edge NLP: train a smaller all-ASSTF variant with patience=1000.

- Replaces the final pool Linear with ASSTFLinear.
- Uses a smaller hidden size (96) and rank=2 to reduce parameters.
- Compares against the previously trained static baseline and the previous ASSTF model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from asstf import ASSTFLinear, count_parameters
from app_06_edge_nlp.train import TinyBertFFN, TinyBertWithASSTF, load_data


class TinyBertAllASSTF(nn.Module):
    """Tiny BERT-style encoder where every controllable linear layer is ASSTF."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 96,
        num_layers: int = 2,
        rank: int = 2,
    ):
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
        # ASSTF-based FFN (already the case in TinyBertWithASSTF).
        self.ffns = nn.ModuleList([
            nn.Sequential(
                ASSTFLinear(hidden_size, hidden_size, structural_rank=rank),
                nn.GELU(),
                ASSTFLinear(hidden_size, hidden_size, structural_rank=rank),
            )
            for _ in range(num_layers)
        ])
        # Final classification layer also replaced by ASSTF.
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
    asstf_patience = 1000

    train_loader, test_loader, vocab_size, use_hf = load_data(batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cp_dir = ROOT / "checkpoints" / "app_06_edge_nlp"
    cp_dir.mkdir(parents=True, exist_ok=True)

    # Previous models for comparison.
    static_model = TinyBertFFN(vocab_size, hidden_size=128, num_layers=2).to(device)
    prev_asstf_model = TinyBertWithASSTF(vocab_size, hidden_size=128, num_layers=2, rank=4).to(device)
    static_model.load_state_dict(torch.load(cp_dir / "static_model.pt", map_location=device))
    prev_asstf_model.load_state_dict(torch.load(cp_dir / "asstf_model.pt", map_location=device))

    # New smaller all-ASSTF model.
    new_asstf_model = TinyBertAllASSTF(vocab_size, hidden_size=96, num_layers=2, rank=2).to(device)
    opt = torch.optim.AdamW(new_asstf_model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    print("=" * 60)
    print("App 06 Edge NLP: all-ASSTF smaller variant (patience=1000)")
    print("=" * 60)
    print(f"Using HuggingFace data: {use_hf}")
    print(f"Vocab size: {vocab_size}")
    print(f"Static params:     {count_parameters(static_model):,}")
    print(f"Prev ASSTF params: {count_parameters(prev_asstf_model):,}")
    print(f"New ASSTF params:  {count_parameters(new_asstf_model):,}")

    # Train new ASSTF model.
    best_val_acc = 0.0
    best_state = None
    no_improve = 0
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(new_asstf_model, opt, train_loader, True, device, loss_fn)
        val_loss, val_acc = run_epoch(new_asstf_model, None, test_loader, False, device, loss_fn)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[New ASSTF] Epoch {epoch+1:03d}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_state = new_asstf_model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
        # patience=1000 effectively disables early stopping within 200 epochs.
        if no_improve >= asstf_patience:
            print(f"[New ASSTF] Early stopping at epoch {epoch+1} (best val acc {best_val_acc:.4f})")
            break

    if best_state is not None:
        new_asstf_model.load_state_dict(best_state)
    torch.save(new_asstf_model.state_dict(), cp_dir / "asstf_model_small.pt")

    # Final evaluation.
    _, static_acc = run_epoch(static_model, None, test_loader, False, device, loss_fn)
    _, prev_asstf_acc = run_epoch(prev_asstf_model, None, test_loader, False, device, loss_fn)
    _, new_asstf_acc = run_epoch(new_asstf_model, None, test_loader, False, device, loss_fn)

    result = {
        "use_hf_data": use_hf,
        "static": {"params": count_parameters(static_model), "test_acc": static_acc},
        "prev_asstf": {"params": count_parameters(prev_asstf_model), "test_acc": prev_asstf_acc},
        "new_asstf": {
            "params": count_parameters(new_asstf_model),
            "test_acc": new_asstf_acc,
            "best_val_acc": best_val_acc,
            "hidden_size": 96,
            "rank": 2,
        },
    }
    out_path = ROOT / "results" / "app_06_edge_nlp" / "experiment_asstf_only.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\nFinal comparison:")
    print(f"  Static (hidden=128): {static_acc:.4f} | params {count_parameters(static_model):,}")
    print(f"  Prev ASSTF (h=128):  {prev_asstf_acc:.4f} | params {count_parameters(prev_asstf_model):,}")
    print(f"  New ASSTF (h=96):    {new_asstf_acc:.4f} | params {count_parameters(new_asstf_model):,}")
    print(f"\nResult written to {out_path}")


if __name__ == "__main__":
    main()
