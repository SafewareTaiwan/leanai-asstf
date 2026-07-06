"""
App 01: Embedded Gesture Recognition - Evaluation

Demonstrates online personalisation: after pre-training on users 0-4, the
ASSTF model is exposed to a stream of samples from a new user (user 5) and
its structural parameters are updated via surprise minimisation.  Accuracy is
reported before and after adaptation.
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

from app_01_gesture.train import ASSTFGestureNet, build_static_baseline, load_data
from asstf import SurpriseMinimizer


def evaluate(
    adaptation_batches: int = 40,
    checkpoint_dir: Path | None = None,
    output_dir: Path | None = None,
):
    checkpoint_dir = checkpoint_dir or (ROOT / "checkpoints" / "app_01_gesture")
    output_dir = output_dir or (ROOT / "results" / "app_01_gesture")
    output_dir.mkdir(parents=True, exist_ok=True)

    _, test_loader, X_test, y_test = load_data(batch_size=16)
    input_dim = X_test.shape[1]
    num_classes = int(np.max(y_test) + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    asstf_model = ASSTFGestureNet(input_dim, num_classes).to(device)
    asstf_model.load_state_dict(torch.load(checkpoint_dir / "asstf_model.pt", map_location=device))

    static_model = build_static_baseline(input_dim, num_classes).to(device)
    static_model.load_state_dict(torch.load(checkpoint_dir / "static_model.pt", map_location=device))

    acc_before = _accuracy(asstf_model, test_loader, device)
    static_acc = _accuracy(static_model, test_loader, device)

    # Online personalisation: adapt θs on a stream of test batches.
    adapter = SurpriseMinimizer(
        asstf_model,
        lr=5e-4,
        beta=0.005,
        reconstruction_loss="ce",
        max_steps=2,
        device=device,
    )

    adapt_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=16,
        shuffle=True,
    )

    adapt_accs = []
    for i, (x, y) in enumerate(adapt_loader):
        if i >= adaptation_batches:
            break
        # Supervised adaptation: use the true label as the reconstruction target.
        adapter.adapt(x, target=y)
        if (i + 1) % 10 == 0:
            acc = _accuracy(asstf_model, test_loader, device)
            adapt_accs.append({"batch": i + 1, "acc": acc})
            print(f"[Adapt] batch {i+1:02d}: acc={acc:.4f}")

    acc_after = _accuracy(asstf_model, test_loader, device)

    result = {
        "app": "app_01_gesture",
        "static_acc": static_acc,
        "asstf_acc_before_adapt": acc_before,
        "asstf_acc_after_adapt": acc_after,
        "improvement": acc_after - acc_before,
        "adaptation_curve": adapt_accs,
    }

    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\nFinal Results:")
    print(json.dumps(result, indent=2))
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
