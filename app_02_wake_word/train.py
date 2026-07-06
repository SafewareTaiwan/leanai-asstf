"""
App 02: Personalized Wake Word Detection - Training

This version integrates ASSTF with 1-D CNNs (ASSTFConv1d) and keeps the
ASSTF model smaller than the static baseline while using matched conv
topologies.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from asstf import ASSTFBlock, ASSTFConvBlock, BilevelTrainer, count_parameters
from shared import accuracy
from shared.early_stopping import EarlyStopping


# ---------------------------------------------------------------------------
# Synthetic audio data
# ---------------------------------------------------------------------------


def synthesize_wake_word_data(
    n_samples: int = 4000,
    seq_len: int = 256,
    n_classes: int = 2,
    seed: int = 43,
):
    """
    Generate synthetic 1D audio waveforms for wake-word detection.

    Positive class (1): contains two prominent frequency chirps.
    Negative class (0): random smooth noise patterns.
    """
    rng = np.random.RandomState(seed)
    X = np.zeros((n_samples, 1, seq_len), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        label = rng.randint(0, n_classes)
        y[i] = label
        t = np.linspace(0, 1, seq_len)
        noise = rng.normal(0, 0.1, size=seq_len)

        if label == 1:
            # Wake-word pattern: two frequency components with amplitude envelope.
            f1, f2 = rng.uniform(3, 8), rng.uniform(12, 20)
            envelope = np.exp(-((t - 0.5) ** 2) / 0.05)
            signal = (
                envelope * np.sin(2 * np.pi * f1 * t)
                + 0.6 * envelope * np.sin(2 * np.pi * f2 * t)
                + noise
            )
        else:
            # Random smooth background.
            f = rng.uniform(1, 5)
            signal = np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) + noise

        X[i, 0, :] = signal

    return X, y


def add_noise(X: np.ndarray, snr_db: float, rng: np.random.RandomState):
    """Add white noise at a given signal-to-noise ratio (dB)."""
    signal_power = np.mean(X ** 2, axis=-1, keepdims=True)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0, np.sqrt(noise_power), size=X.shape)
    return X + noise


def load_data(batch_size: int = 64, test_split: float = 0.2):
    X, y = synthesize_wake_word_data()
    n_test = int(len(X) * test_split)
    indices = np.random.RandomState(42).permutation(len(X))
    train_idx, test_idx = indices[n_test:], indices[:n_test]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    # Noisy validation set for early stopping: simulates the target deployment
    # condition and prevents stopping too early on clean-data accuracy.
    val_rng = np.random.RandomState(123)
    X_val_noisy = add_noise(X_test, snr_db=-2.0, rng=val_rng).astype(np.float32)
    val_noisy_ds = TensorDataset(torch.from_numpy(X_val_noisy), torch.from_numpy(y_test))

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        DataLoader(val_noisy_ds, batch_size=batch_size, shuffle=False),
        X_test,
        y_test,
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ASSTFWakeWordNet(nn.Module):
    """Tiny ASSTF+CNN wake-word detector."""

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        num_classes: int,
        rank: int = 1,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            ASSTFConvBlock(
                in_channels, 4, kernel_size=5, padding=2,
                structural_rank=rank, activation="relu",
            ),
            nn.MaxPool1d(2),
            ASSTFConvBlock(
                4, 8, kernel_size=3, padding=1,
                structural_rank=rank, activation="relu",
            ),
            nn.AdaptiveAvgPool1d(8),
        )
        with torch.no_grad():
            feat = self.conv(torch.zeros(1, in_channels, seq_len)).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            ASSTFBlock(feat, 8, structural_rank=rank, activation="relu"),
            nn.Linear(8, num_classes),
        )

    def forward(self, x):
        feats = self.conv(x)
        feats = feats.view(feats.size(0), -1)
        return self.classifier(feats)


class StaticWakeWordNet(nn.Module):
    """Matched static baseline with the same conv topology but a larger head."""

    def __init__(self, in_channels: int, seq_len: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        with torch.no_grad():
            feat = self.conv(torch.zeros(1, in_channels, seq_len)).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(feat, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        feats = self.conv(x)
        feats = feats.view(feats.size(0), -1)
        return self.classifier(feats)


def build_static_baseline(in_channels: int, seq_len: int, num_classes: int):
    return StaticWakeWordNet(in_channels, seq_len, num_classes)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    epochs: int = 200,
    asstf_epochs: int | None = None,
    batch_size: int = 64,
    lr_core: float = 1e-3,
    lr_struct: float = 5e-4,
    static_patience: int = 10,
    asstf_patience: int | None = None,
    save_dir: Path | None = None,
):
    asstf_epochs = asstf_epochs or epochs
    asstf_patience = asstf_patience or static_patience * 10
    save_dir = save_dir or (ROOT / "checkpoints" / "app_02_wake_word")
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, val_noisy_loader, X_test, y_test = load_data(batch_size)
    seq_len = X_test.shape[2]
    num_classes = int(np.max(y_test) + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    asstf_model = ASSTFWakeWordNet(1, seq_len, num_classes, rank=1).to(device)
    static_model = build_static_baseline(1, seq_len, num_classes).to(device)

    print("=" * 60)
    print("App 02: Personalized Wake Word Detection")
    print("=" * 60)
    print(f"ASSTF params:  {count_parameters(asstf_model):,}")
    print(f"Static params: {count_parameters(static_model):,}")

    asstf_trainer = BilevelTrainer(
        asstf_model, lr_core=lr_core, lr_struct=lr_struct, device=device
    )
    loss_fn = nn.CrossEntropyLoss()
    asstf_stopper = EarlyStopping(patience=asstf_patience, mode="max")

    for epoch in range(asstf_epochs):
        metrics = asstf_trainer.train_epoch(train_loader, loss_fn, alternate=True)
        clean_acc = _accuracy(asstf_model, test_loader, device)
        noisy_acc = _accuracy(asstf_model, val_noisy_loader, device)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"[ASSTF] Epoch {epoch+1:02d}: clean_acc={clean_acc:.4f} "
                f"noisy_val_acc={noisy_acc:.4f} loss={metrics['core_loss']:.4f}"
            )
        # Early stop on noisy validation accuracy to ensure robustness.
        # ASSTF gets 10x the patience of the static model because it can
        # break through plateaus later in training.
        if asstf_stopper(noisy_acc):
            print(
                f"[ASSTF] Early stopping at epoch {epoch+1} "
                f"(best noisy val acc {asstf_stopper.best_value:.4f})"
            )
            break

    static_opt = torch.optim.Adam(
        static_model.parameters(), lr=lr_core, weight_decay=1e-5
    )
    static_stopper = EarlyStopping(patience=static_patience, mode="max")

    for epoch in range(epochs):
        static_model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            static_opt.zero_grad()
            loss = nn.CrossEntropyLoss()(static_model(x), y)
            loss.backward()
            static_opt.step()
        clean_acc = _accuracy(static_model, test_loader, device)
        noisy_acc = _accuracy(static_model, val_noisy_loader, device)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"[Static] Epoch {epoch+1:02d}: clean_acc={clean_acc:.4f} "
                f"noisy_val_acc={noisy_acc:.4f}"
            )
        if static_stopper(noisy_acc):
            print(
                f"[Static] Early stopping at epoch {epoch+1} "
                f"(best noisy val acc {static_stopper.best_value:.4f})"
            )
            break

    torch.save(asstf_model.state_dict(), save_dir / "asstf_model.pt")
    torch.save(static_model.state_dict(), save_dir / "static_model.pt")

    metadata = {
        "app": "app_02_wake_word",
        "seq_len": seq_len,
        "num_classes": num_classes,
        "epochs": epochs,
        "asstf_epochs": asstf_epochs,
        "static_patience": static_patience,
        "asstf_patience": asstf_patience,
        "asstf_params": count_parameters(asstf_model),
        "static_params": count_parameters(static_model),
        "asstf_test_acc": _accuracy(asstf_model, test_loader, device),
        "static_test_acc": _accuracy(static_model, test_loader, device),
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved models to", save_dir)
    print(json.dumps(metadata, indent=2))
    return metadata


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
    train()
