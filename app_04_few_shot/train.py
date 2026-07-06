"""
App 04: Few-Shot Meta-Learner - Training
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from asstf import ASSTFBlock, BilevelTrainer, count_parameters
from shared import StaticMLP, accuracy
from shared.early_stopping import EarlyStopping


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_digits_data():
    """Load sklearn digits as a pool for few-shot tasks."""
    digits = load_digits()
    X = digits.images.astype(np.float32) / 16.0  # normalize to [0,1]
    X = X[:, None, :, :]  # add channel dim: (B, 1, 8, 8)
    y = digits.target.astype(np.int64)
    return X, y


def sample_few_shot_task(X, y, n_way=5, k_shot=1, q_query=15, rng=None):
    """Sample one N-way K-shot task."""
    rng = rng or np.random.RandomState()
    classes = rng.choice(np.unique(y), size=n_way, replace=False)
    support_X, support_y, query_X, query_y = [], [], [], []

    for new_label, cls in enumerate(classes):
        idx = np.where(y == cls)[0]
        selected = rng.choice(idx, size=k_shot + q_query, replace=False)
        support_X.append(X[selected[:k_shot]])
        support_y.append(np.full(k_shot, new_label))
        query_X.append(X[selected[k_shot:]])
        query_y.append(np.full(q_query, new_label))

    support_X = np.concatenate(support_X, axis=0)
    support_y = np.concatenate(support_y, axis=0)
    query_X = np.concatenate(query_X, axis=0)
    query_y = np.concatenate(query_y, axis=0)

    perm = rng.permutation(len(support_X))
    support_X, support_y = support_X[perm], support_y[perm]
    perm = rng.permutation(len(query_X))
    query_X, query_y = query_X[perm], query_y[perm]

    return (
        torch.from_numpy(support_X),
        torch.from_numpy(support_y),
        torch.from_numpy(query_X),
        torch.from_numpy(query_y),
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ASSTFFewShotNet(nn.Module):
    def __init__(self, n_way: int, rank: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            ASSTFBlock(16, 32, structural_rank=rank, activation="relu"),
            nn.Linear(32, n_way),
        )

    def forward(self, x):
        feats = self.features(x).view(x.size(0), -1)
        return self.classifier(feats)


class StaticFewShotNet(nn.Module):
    def __init__(self, n_way: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, n_way),
        )

    def forward(self, x):
        feats = self.features(x).view(x.size(0), -1)
        return self.classifier(feats)


def build_static_baseline(n_way: int):
    return StaticFewShotNet(n_way)


# ---------------------------------------------------------------------------
# Training (meta-training)
# ---------------------------------------------------------------------------

def train(
    n_way: int = 5,
    k_shot: int = 1,
    q_query: int = 15,
    n_tasks: int = 2000,
    asstf_tasks: int | None = None,
    inner_steps: int = 5,
    lr_core: float = 1e-3,
    lr_struct: float = 5e-4,
    static_patience: int = 200,
    asstf_patience: int | None = None,
    val_interval: int = 50,
    save_dir: Path | None = None,
):
    asstf_tasks = asstf_tasks or n_tasks
    asstf_patience = asstf_patience or static_patience * 10
    save_dir = save_dir or (ROOT / "checkpoints" / "app_04_few_shot")
    save_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_digits_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    asstf_model = ASSTFFewShotNet(n_way, rank=3).to(device)
    static_model = build_static_baseline(n_way).to(device)

    print("=" * 60)
    print("App 04: Few-Shot Meta-Learner")
    print("=" * 60)
    print(f"ASSTF params:  {count_parameters(asstf_model):,}")
    print(f"Static params: {count_parameters(static_model):,}")

    asstf_trainer = BilevelTrainer(
        asstf_model, lr_core=lr_core, lr_struct=lr_struct, device=device
    )
    static_opt = torch.optim.Adam(static_model.parameters(), lr=lr_core, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    rng = np.random.RandomState(46)
    val_rng = np.random.RandomState(999)

    def val_acc(model, n_val=20):
        model.eval()
        accs = []
        with torch.no_grad():
            for _ in range(n_val):
                s_x, s_y, q_x, q_y = sample_few_shot_task(X, y, n_way, k_shot, q_query, val_rng)
                s_x, s_y = s_x.to(device), s_y.to(device)
                q_x, q_y = q_x.to(device), q_y.to(device)
                accs.append(accuracy(model(q_x), q_y))
        return float(np.mean(accs))

    static_stopper = EarlyStopping(patience=static_patience, mode="max")
    # Train static baseline until validation accuracy plateaus.
    for task_id in range(n_tasks):
        s_x, s_y, q_x, q_y = sample_few_shot_task(X, y, n_way, k_shot, q_query, rng)
        s_x, s_y = s_x.to(device), s_y.to(device)

        static_model.train()
        static_opt.zero_grad()
        loss_s = loss_fn(static_model(s_x), s_y)
        loss_s.backward()
        static_opt.step()

        if (task_id + 1) % val_interval == 0:
            vacc = val_acc(static_model)
            print(f"[Static] Task {task_id+1:03d}: val acc={vacc:.4f}")
            if static_stopper(vacc):
                print(f"[Static] Early stopping at task {task_id+1} (best val acc {static_stopper.best_value:.4f})")
                break

    asstf_stopper = EarlyStopping(patience=asstf_patience, mode="max")

    def _train_task_alternating(support_x, support_y):
        """Run one alternating core / structural update on a support set."""
        batch = (support_x, support_y)
        c_loss = asstf_trainer._phase_step(loss_fn, batch, phase="core")
        s_loss = asstf_trainer._phase_step(loss_fn, batch, phase="struct")
        return c_loss, s_loss

    # Train ASSTF until validation accuracy plateaus.
    rng2 = np.random.RandomState(46)
    for task_id in range(asstf_tasks):
        s_x, s_y, q_x, q_y = sample_few_shot_task(X, y, n_way, k_shot, q_query, rng2)
        s_x, s_y = s_x.to(device), s_y.to(device)

        asstf_model.train()
        _train_task_alternating(s_x, s_y)

        if (task_id + 1) % val_interval == 0:
            vacc = val_acc(asstf_model)
            print(f"[ASSTF] Task {task_id+1:03d}: val acc={vacc:.4f}")
            if asstf_stopper(vacc):
                print(f"[ASSTF] Early stopping at task {task_id+1} (best val acc {asstf_stopper.best_value:.4f})")
                break

    torch.save(asstf_model.state_dict(), save_dir / "asstf_model.pt")
    torch.save(static_model.state_dict(), save_dir / "static_model.pt")

    metadata = {
        "app": "app_04_few_shot",
        "n_way": n_way,
        "k_shot": k_shot,
        "q_query": q_query,
        "n_tasks": n_tasks,
        "asstf_tasks": asstf_tasks,
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
