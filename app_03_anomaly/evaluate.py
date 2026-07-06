"""
App 03: Time Series Anomaly Detection - Evaluation

Simulate a 30-day deployment.  The test stream is divided into 10 sequential
windows; windows 1-7 follow the training distribution, windows 8-10 contain
concept drift.  The ASSTF model performs online structural adaptation via
surprise minimisation (Eq. 14-15) on incoming *normal* windows, while the
static model remains frozen.  We report per-window F1-score.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app_03_anomaly.train import ASSTFAutoencoder, build_static_baseline, load_data
from asstf import SurpriseMinimizer
from sklearn.metrics import f1_score


def evaluate(
    n_windows: int = 10,
    threshold_percentile: float = 99.0,
    threshold_max_multiplier: float = 2.0,
    adapt_multiplier: float = 1.5,
    threshold_margin: float = 1.1,
    checkpoint_dir: Path | None = None,
    output_dir: Path | None = None,
):
    checkpoint_dir = checkpoint_dir or (ROOT / "checkpoints" / "app_03_anomaly")
    output_dir = output_dir or (ROOT / "results" / "app_03_anomaly")
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, X_test, y_test = load_data()
    input_dim = X_test.shape[2]
    seq_len = X_test.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    asstf_model = ASSTFAutoencoder(input_dim, seq_len, latent_dim=1, hidden_dim=5, rank=1).to(device)
    asstf_model.load_state_dict(torch.load(checkpoint_dir / "asstf_model.pt", map_location=device))

    static_model = build_static_baseline(input_dim, seq_len).to(device)
    static_model.load_state_dict(torch.load(checkpoint_dir / "static_model.pt", map_location=device))

    # Split test set into sequential windows.
    window_size = len(X_test) // n_windows
    windows_X = [X_test[i * window_size : (i + 1) * window_size] for i in range(n_windows)]
    windows_y = [y_test[i * window_size : (i + 1) * window_size] for i in range(n_windows)]

    def threshold_for(model, X_win):
        model.eval()
        with torch.no_grad():
            recon = model(torch.from_numpy(X_win).to(device)).cpu().numpy()
        errs = np.mean((recon - X_win) ** 2, axis=(1, 2))
        return np.percentile(errs, threshold_percentile)

    asstf_threshold = threshold_for(asstf_model, windows_X[0])
    static_threshold = threshold_for(static_model, windows_X[0])
    initial_asstf_threshold = float(asstf_threshold)
    max_asstf_threshold = initial_asstf_threshold * threshold_max_multiplier

    # Structural adapter: Eq. (14)-(15) surprise minimisation on θs.
    adapter = SurpriseMinimizer(
        asstf_model,
        lr=1e-3,
        beta=0.001,
        reconstruction_loss="mse",
        max_steps=5,
        device=device,
    )

    # Running pool of normal-window errors used to refresh the ASSTF threshold.
    normal_errors = []

    asstf_f1s, static_f1s = [], []
    asstf_errs, static_errs = [], []
    asstf_all_pred, asstf_all_y = [], []
    static_all_pred, static_all_y = [], []
    adapted_windows = 0
    for i, (wx, wy) in enumerate(zip(windows_X, windows_y)):
        wx_t = torch.from_numpy(wx).to(device)

        # Estimate the surprise for this window before any adaptation.
        cur_errs = _reconstruction_error(asstf_model, wx_t)
        window_mean_err = float(np.mean(cur_errs))

        # Only adapt on windows that look normal (low surprise).  Anomalous
        # windows have high reconstruction error and must not pollute θs.
        adapt_threshold = asstf_threshold * adapt_multiplier
        window_max_err = float(np.max(cur_errs))
        # A window is treated as normal only if both its mean and max errors
        # are moderate.  The absolute max-error guard prevents the threshold
        # from being inflated by windows that contain obvious anomalies.
        adapted = (window_mean_err <= adapt_threshold) and (window_max_err <= 1.0)
        if adapted:
            adapter.adapt(wx_t, target=wx_t)
            adapted_windows += 1
            # Re-evaluate after adaptation for scoring.
            cur_errs = _reconstruction_error(asstf_model, wx_t)
            window_mean_err = float(np.mean(cur_errs))
            # Refresh the normal-error pool and the decision threshold online.
            normal_errors.extend(cur_errs.tolist())
            asstf_threshold = float(
                np.percentile(normal_errors, threshold_percentile) * threshold_margin
            )
            asstf_threshold = min(asstf_threshold, max_asstf_threshold)

        # Score both models.
        asstf_scores = cur_errs
        static_scores = _reconstruction_error(static_model, torch.from_numpy(wx).to(device))

        asstf_pred = (asstf_scores > asstf_threshold).astype(np.int64)
        static_pred = (static_scores > static_threshold).astype(np.int64)

        asstf_f1s.append(float(f1_score(wy, asstf_pred, pos_label=1, zero_division=0)))
        static_f1s.append(float(f1_score(wy, static_pred, pos_label=1, zero_division=0)))
        asstf_errs.append(window_mean_err)
        static_errs.append(float(np.mean(static_scores)))
        asstf_all_pred.append(asstf_pred)
        asstf_all_y.append(wy)
        static_all_pred.append(static_pred)
        static_all_y.append(wy)
        print(
            f"Window {i+1:02d}: ASSTF F1={asstf_f1s[-1]:.4f} "
            f"| Static F1={static_f1s[-1]:.4f} "
            f"| adapted={adapted}"
        )

    global_f1_asstf = float(
        f1_score(np.concatenate(asstf_all_y), np.concatenate(asstf_all_pred), pos_label=1, zero_division=0)
    )
    global_f1_static = float(
        f1_score(np.concatenate(static_all_y), np.concatenate(static_all_pred), pos_label=1, zero_division=0)
    )

    result = {
        "app": "app_03_anomaly",
        "asstf_threshold": float(asstf_threshold),
        "static_threshold": float(static_threshold),
        "adapt_threshold": float(adapt_threshold),
        "adapted_windows": adapted_windows,
        "window_f1_asstf": asstf_f1s,
        "window_f1_static": static_f1s,
        "mean_f1_asstf": float(np.mean(asstf_f1s)),
        "mean_f1_static": float(np.mean(static_f1s)),
        "global_f1_asstf": global_f1_asstf,
        "global_f1_static": global_f1_static,
        "window_recon_error_asstf": asstf_errs,
        "window_recon_error_static": static_errs,
        "mean_recon_error_asstf": float(np.mean(asstf_errs)),
        "mean_recon_error_static": float(np.mean(static_errs)),
        "final_asstf_threshold": float(asstf_threshold),
    }
    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n", json.dumps(result, indent=2))
    return result


def _reconstruction_error(model, x):
    """Per-sample anomaly score = mean reconstruction error over (time, dim)."""
    model.eval()
    with torch.no_grad():
        recon = model(x).cpu().numpy()
    x_np = x.cpu().numpy()
    return np.mean((recon - x_np) ** 2, axis=(1, 2))


if __name__ == "__main__":
    evaluate()
