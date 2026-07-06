"""
App 05: Online RL Dynamic Policy - Evaluation

Evaluate the pre-trained policies before and after a sudden environmental
perturbation (gravity +50%).  ASSTF adapts its structural parameters online
using the reward signal as the adaptation objective.
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

from app_05_rl.train import (
    ASSTFPolicy,
    SimplePendulumEnv,
    build_static_baseline,
    collect_episode,
    compute_returns,
)
from asstf import SurpriseMinimizer


def evaluate(
    n_eval_episodes: int = 30,
    adaptation_steps: int = 20,
    checkpoint_dir: Path | None = None,
    output_dir: Path | None = None,
):
    checkpoint_dir = checkpoint_dir or (ROOT / "checkpoints" / "app_05_rl")
    output_dir = output_dir or (ROOT / "results" / "app_05_rl")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    asstf_policy = ASSTFPolicy(rank=3).to(device)
    asstf_policy.load_state_dict(torch.load(checkpoint_dir / "asstf_model.pt", map_location=device))

    static_policy = build_static_baseline().to(device)
    static_policy.load_state_dict(torch.load(checkpoint_dir / "static_model.pt", map_location=device))

    rng = np.random.RandomState(123)

    def rollout_reward(policy, gravity, n_eps=10):
        env = SimplePendulumEnv(gravity=gravity)
        returns = []
        for _ in range(n_eps):
            _, _, rewards = collect_episode(env, policy, device, rng)
            returns.append(float(np.sum(rewards)))
        return float(np.mean(returns)), float(np.std(returns))

    # Baseline gravity.
    asstf_normal_mean, asstf_normal_std = rollout_reward(asstf_policy, gravity=9.8, n_eps=n_eval_episodes // 3)
    static_normal_mean, static_normal_std = rollout_reward(static_policy, gravity=9.8, n_eps=n_eval_episodes // 3)

    print(f"Normal gravity 9.8 | ASSTF {asstf_normal_mean:.2f}±{asstf_normal_std:.2f} | Static {static_normal_mean:.2f}±{static_normal_std:.2f}")

    # Perturbed gravity.
    perturbed_gravity = 9.8 * 1.5
    asstf_perturb_mean, asstf_perturb_std = rollout_reward(asstf_policy, gravity=perturbed_gravity, n_eps=n_eval_episodes // 3)
    static_perturb_mean, static_perturb_std = rollout_reward(static_policy, gravity=perturbed_gravity, n_eps=n_eval_episodes // 3)

    print(f"Perturbed gravity {perturbed_gravity:.1f} | ASSTF {asstf_perturb_mean:.2f}±{asstf_perturb_std:.2f} | Static {static_perturb_mean:.2f}±{static_perturb_std:.2f}")

    # Online adaptation for ASSTF.
    # Only structural parameters are updated; the objective is a policy-gradient
    # signal computed on episodes collected under the perturbed dynamics.
    adapter = SurpriseMinimizer(
        asstf_policy,
        lr=1e-3,
        beta=0.001,
        reconstruction_loss="mse",
        max_steps=1,
        device=device,
    )

    env = SimplePendulumEnv(gravity=perturbed_gravity)
    gamma = 0.99
    for step in range(adaptation_steps):
        states, actions, rewards = collect_episode(env, asstf_policy, device, rng)
        returns = compute_returns(rewards, gamma)

        s_t = torch.from_numpy(states).float().to(device)
        a_t = torch.from_numpy(actions).float().unsqueeze(-1).to(device)
        r_t = torch.from_numpy(returns).float().unsqueeze(-1).to(device)
        r_t = (r_t - r_t.mean()) / (r_t.std() + 1e-8)

        def pg_loss(out, _):
            log_prob = -((out - a_t) ** 2)
            return -(log_prob * r_t).mean()

        adapter.adapt(s_t, loss_fn=pg_loss)
        if (step + 1) % 5 == 0:
            mean_r, std_r = rollout_reward(asstf_policy, gravity=perturbed_gravity, n_eps=3)
            print(f"[ASSTF adapt] step {step+1}: reward={mean_r:.2f}±{std_r:.2f}")

    asstf_adapted_mean, asstf_adapted_std = rollout_reward(asstf_policy, gravity=perturbed_gravity, n_eps=n_eval_episodes // 3)

    result = {
        "app": "app_05_rl",
        "gravity_normal": 9.8,
        "gravity_perturbed": perturbed_gravity,
        "asstf_normal_reward": asstf_normal_mean,
        "static_normal_reward": static_normal_mean,
        "asstf_perturbed_reward": asstf_perturb_mean,
        "static_perturbed_reward": static_perturb_mean,
        "asstf_adapted_reward": asstf_adapted_mean,
    }
    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nFinal:", json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    evaluate()
