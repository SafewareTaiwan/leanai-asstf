"""
Grid search for the smallest ASSTF gesture model that reaches 100% test accuracy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app_01_gesture.train import load_data
from asstf import ASSTFBlock, BilevelTrainer, count_parameters


def build_model(input_dim: int, num_classes: int, h1: int, h2: int, rank: int):
    return nn.Sequential(
        ASSTFBlock(input_dim, h1, structural_rank=rank, activation="relu"),
        ASSTFBlock(h1, h2, structural_rank=rank, activation="relu"),
        nn.Linear(h2, num_classes),
    )


def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def train_and_eval(h1: int, h2: int, rank: int, max_epochs: int = 100, patience: int = 20):
    train_loader, test_loader, X_test, y_test = load_data(batch_size=64)
    input_dim = X_test.shape[1]
    num_classes = int(np.max(y_test) + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(input_dim, num_classes, h1, h2, rank).to(device)
    trainer = BilevelTrainer(model, lr_core=1e-3, lr_struct=5e-4, device=device)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    no_improve = 0
    for epoch in range(max_epochs):
        trainer.train_epoch(train_loader, loss_fn, alternate=True)
        acc = evaluate_accuracy(model, test_loader, device)
        if acc > best_acc + 1e-6:
            best_acc = acc
            no_improve = 0
        else:
            no_improve += 1
        if best_acc >= 1.0 or no_improve >= patience:
            break

    params = count_parameters(model)
    return params, best_acc, epoch + 1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    candidates = []
    # Search small hidden sizes and ranks.
    for h1 in [4, 6, 8, 10, 12, 16, 20, 24, 32]:
        for h2 in [2, 4, 6, 8, 10, 12, 16, 20, 24]:
            if h2 >= h1:
                continue
            for rank in [1, 2, 3]:
                if rank > min(h1, h2):
                    continue
                candidates.append((h1, h2, rank))

    results = []
    for h1, h2, rank in candidates:
        params, acc, epochs = train_and_eval(h1, h2, rank)
        status = "OK" if acc >= 1.0 else "FAIL"
        print(f"h1={h1:3d} h2={h2:3d} rank={rank} params={params:6d} acc={acc:.4f} epochs={epochs:3d} [{status}]")
        results.append((params, acc, h1, h2, rank, epochs))

    ok = [r for r in results if r[1] >= 1.0]
    if not ok:
        print("\nNo model reached 100% accuracy in this search space.")
        return

    ok.sort(key=lambda r: r[0])
    print("\nTop 10 smallest models reaching 100% accuracy:")
    for params, acc, h1, h2, rank, epochs in ok[:10]:
        print(f"  params={params:6d} h1={h1:3d} h2={h2:3d} rank={rank} epochs={epochs:3d} acc={acc:.4f}")

    best = ok[0]
    print(f"\nMinimum ASSTF model achieving 100%: h1={best[2]}, h2={best[3]}, rank={best[4]}, params={best[0]}")


if __name__ == "__main__":
    main()
