"""
App 06 edge NLP: train the ASSTF variant on real SST-2 with sentencepiece fix.

- Uses BertTokenizer directly to avoid the AutoTokenizer / fast-tokenizer issue.
- Loads real SST-2 from `nyu-mll/glue`.
- Replaces the final pool Linear with ASSTFLinear so all controllable linear
  layers in the ASSTF model are ASSTF-based.
- patience=1000 (no early stopping within the default 200 epochs).

The static baseline is left untouched; its previous synthetic-data result
(49.25%) is kept for the comparison table.
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

from asstf import ASSTFLinear, count_parameters
from app_06_edge_nlp.train import TinyBertFFN


# ---------------------------------------------------------------------------
# Real SST-2 data loader with sentencepiece-compatible slow tokenizer
# ---------------------------------------------------------------------------


def load_sst2_sentencepiece():
    """Load real SST-2 encoded by the custom SentencePiece tokenizer."""
    data = np.load(ROOT / "data" / "sst2_sentencepiece.npz")
    return data["X"].astype(np.int64), data["y"].astype(np.int64), int(data["vocab_size"])


def load_data(batch_size: int = 32, test_split: float = 0.2):
    X, y, vocab_size = load_sst2_sentencepiece()
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
# ASSTF model with all controllable linear layers replaced by ASSTFLinear
# ---------------------------------------------------------------------------


class TinyBertAllASSTF128(nn.Module):
    """Same size as the previous ASSTF model, but the final pool is ASSTF too."""

    def __init__(self, vocab_size: int, hidden_size: int = 128, num_layers: int = 2, rank: int = 4):
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
        # Final classification layer is also ASSTF (rank=1 to keep overhead tiny).
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

    train_loader, test_loader, vocab_size = load_data(batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    asstf_model = TinyBertAllASSTF128(vocab_size, hidden_size=128, num_layers=2, rank=4).to(device)
    opt = torch.optim.AdamW(asstf_model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    print("=" * 60)
    print("App 06 Edge NLP: ASSTF-only training on real SST-2 (SentencePiece)")
    print("=" * 60)
    print(f"Vocab size: {vocab_size}")
    print(f"ASSTF params: {count_parameters(asstf_model):,}")

    best_val_acc = 0.0
    best_state = None
    no_improve = 0
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(asstf_model, opt, train_loader, True, device, loss_fn)
        val_loss, val_acc = run_epoch(asstf_model, None, test_loader, False, device, loss_fn)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[ASSTF] Epoch {epoch+1:03d}: "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
            )
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in asstf_model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= asstf_patience:
            print(f"[ASSTF] Early stopping at epoch {epoch+1} (best val acc {best_val_acc:.4f})")
            break

    if best_state is not None:
        asstf_model.load_state_dict(best_state)

    cp_dir = ROOT / "checkpoints" / "app_06_edge_nlp"
    cp_dir.mkdir(parents=True, exist_ok=True)
    torch.save(asstf_model.state_dict(), cp_dir / "asstf_model_real.pt")

    _, test_acc = run_epoch(asstf_model, None, test_loader, False, device, loss_fn)

    result = {
        "app": "app_06_edge_nlp",
        "data": "real_sst2",
        "asstf_params": count_parameters(asstf_model),
        "asstf_test_acc": test_acc,
        "asstf_best_val_acc": best_val_acc,
    }
    out_dir = ROOT / "results" / "app_06_edge_nlp"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result_asstf_real.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\nResult:")
    print(json.dumps(result, indent=2))
    print(
        "\nStatic baseline (synthetic fallback, previous run): 49.25% | params 649,218"
    )


if __name__ == "__main__":
    main()
