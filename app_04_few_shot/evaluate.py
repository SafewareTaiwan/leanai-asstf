"""
App 04: Few-Shot Meta-Learner - Evaluation

Evaluate on held-out few-shot tasks.  For ASSTF, we freeze core parameters
and adapt only structural parameters on the support set, demonstrating that
θs acts as task-specific meta-knowledge.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app_04_few_shot.train import ASSTFFewShotNet, build_static_baseline, load_digits_data, sample_few_shot_task
from asstf import SurpriseMinimizer, count_parameters
from shared import accuracy


def evaluate(
    n_way: int = 5,
    k_shot: int = 1,
    q_query: int = 15,
    n_eval_tasks: int = 100,
    inner_steps: int = 10,
    checkpoint_dir: Path | None = None,
    output_dir: Path | None = None,
):
    checkpoint_dir = checkpoint_dir or (ROOT / "checkpoints" / "app_04_few_shot")
    output_dir = output_dir or (ROOT / "results" / "app_04_few_shot")
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_digits_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    asstf_model = ASSTFFewShotNet(n_way, rank=3).to(device)
    asstf_model.load_state_dict(torch.load(checkpoint_dir / "asstf_model.pt", map_location=device))

    static_model = build_static_baseline(n_way).to(device)
    static_model.load_state_dict(torch.load(checkpoint_dir / "static_model.pt", map_location=device))

    rng = np.random.RandomState(999)
    loss_fn = nn.CrossEntropyLoss()

    asstf_accs, static_accs = [], []

    for task_id in range(n_eval_tasks):
        s_x, s_y, q_x, q_y = sample_few_shot_task(X, y, n_way, k_shot, q_query, rng)
        s_x, s_y = s_x.to(device), s_y.to(device)
        q_x, q_y = q_x.to(device), q_y.to(device)

        # Static: freeze feature extractor, fine-tune only classifier on support set.
        static_copy = build_static_baseline(n_way).to(device)
        static_copy.load_state_dict(static_model.state_dict())
        for p in static_copy.features.parameters():
            p.requires_grad = False
        opt = torch.optim.SGD(static_copy.classifier.parameters(), lr=1e-2, momentum=0.9)
        static_copy.train()
        for _ in range(inner_steps):
            opt.zero_grad()
            loss = loss_fn(static_copy(s_x), s_y)
            loss.backward()
            opt.step()
        static_copy.eval()
        with torch.no_grad():
            static_accs.append(accuracy(static_copy(q_x), q_y))

        # ASSTF: freeze core parameters, adapt structural parameters only.
        asstf_copy = ASSTFFewShotNet(n_way, rank=3).to(device)
        asstf_copy.load_state_dict(asstf_model.state_dict())
        for name, p in asstf_copy.named_parameters():
            if not ("structural" in name or "alpha" in name or "zeta" in name or "beta" in name or "gate" in name):
                p.requires_grad = False
        adapter = SurpriseMinimizer(
            asstf_copy,
            lr=5e-3,
            beta=0.001,
            reconstruction_loss="ce",
            max_steps=inner_steps,
            device=device,
        )
        adapter.adapt(s_x, target=s_y, loss_fn=loss_fn)
        asstf_copy.eval()
        with torch.no_grad():
            asstf_accs.append(accuracy(asstf_copy(q_x), q_y))

        if (task_id + 1) % 20 == 0:
            print(
                f"Task {task_id+1:03d}: "
                f"ASSTF={np.mean(asstf_accs[-20:]):.4f} "
                f"Static={np.mean(static_accs[-20:]):.4f}"
            )

    result = {
        "app": "app_04_few_shot",
        "n_eval_tasks": n_eval_tasks,
        "n_way": n_way,
        "k_shot": k_shot,
        "asstf_mean_acc": float(np.mean(asstf_accs)),
        "static_mean_acc": float(np.mean(static_accs)),
        "asstf_std_acc": float(np.std(asstf_accs)),
        "static_std_acc": float(np.std(static_accs)),
    }
    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nFinal:", json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    evaluate()
