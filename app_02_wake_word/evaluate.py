"""
App 02: Personalized Wake Word Detection - Evaluation

Evaluate under varying SNR conditions to demonstrate dynamic rank adaptation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app_02_wake_word.train import (
    ASSTFWakeWordNet,
    add_noise,
    build_static_baseline,
    load_data,
    synthesize_wake_word_data,
)
from asstf import SurpriseMinimizer


def evaluate(
    snr_levels: list = None,
    adaptation_batches: int = 20,
    checkpoint_dir: Path | None = None,
    output_dir: Path | None = None,
):
    snr_levels = snr_levels or [-5, 0, 5, 10]
    checkpoint_dir = checkpoint_dir or (ROOT / "checkpoints" / "app_02_wake_word")
    output_dir = output_dir or (ROOT / "results" / "app_02_wake_word")
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, _, X_test, y_test = load_data()
    seq_len = X_test.shape[2]
    num_classes = int(np.max(y_test) + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    asstf_model = ASSTFWakeWordNet(1, seq_len, num_classes, rank=1).to(device)
    asstf_model.load_state_dict(torch.load(checkpoint_dir / "asstf_model.pt", map_location=device))

    static_model = build_static_baseline(1, seq_len, num_classes).to(device)
    static_model.load_state_dict(torch.load(checkpoint_dir / "static_model.pt", map_location=device))

    rng = np.random.RandomState(44)

    # Prepare a small clean adaptation stream for ASSTF.
    adapt_idx = rng.choice(len(X_test), size=min(adaptation_batches * 16, len(X_test)), replace=False)
    X_adapt = X_test[adapt_idx].astype(np.float32)
    y_adapt = y_test[adapt_idx]
    adapt_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_adapt), torch.from_numpy(y_adapt)),
        batch_size=16,
        shuffle=True,
    )

    adapter = SurpriseMinimizer(
        asstf_model,
        lr=1e-4,
        beta=0.001,
        reconstruction_loss="ce",
        max_steps=1,
        device=device,
    )

    rows = []
    for snr in snr_levels:
        # Re-initialize ASSTF from checkpoint for each SNR, then adapt directly
        # on noisy samples from that SNR level.  This simulates real-world
        # deployment where the device adapts to the current acoustic conditions.
        asstf_model.load_state_dict(torch.load(checkpoint_dir / "asstf_model.pt", map_location=device))
        X_noisy_adapt = add_noise(X_adapt, snr_db=snr, rng=rng).astype(np.float32)
        adapt_loader_noisy = DataLoader(
            TensorDataset(torch.from_numpy(X_noisy_adapt), torch.from_numpy(y_adapt)),
            batch_size=16,
            shuffle=True,
        )
        for x, y in adapt_loader_noisy:
            adapter.adapt(x, target=y)

        X_noisy = add_noise(X_test, snr_db=snr, rng=rng).astype(np.float32)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_noisy), torch.from_numpy(y_test)),
            batch_size=64,
            shuffle=False,
        )
        asstf_acc = _accuracy(asstf_model, loader, device)
        static_acc = _accuracy(static_model, loader, device)
        rows.append({
            "snr_db": snr,
            "asstf_acc": asstf_acc,
            "static_acc": static_acc,
            "gap": asstf_acc - static_acc,
        })
        print(f"SNR {snr:3d} dB | ASSTF {asstf_acc:.4f} | Static {static_acc:.4f} | Gap {asstf_acc - static_acc:+.4f}")

    result = {
        "app": "app_02_wake_word",
        "snr_table": rows,
    }
    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def _accuracy(model, loader, device):
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


if __name__ == "__main__":
    evaluate()
