"""
App 05: Online RL Dynamic Policy - Training
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from asstf import ASSTFBlock, BilevelTrainer, count_parameters
from shared import StaticMLP
from shared.early_stopping import EarlyStopping


# ---------------------------------------------------------------------------
# Custom continuous-control environment
# ---------------------------------------------------------------------------

class SimplePendulumEnv:
    """
    Simplified continuous pendulum-on-cart environment.

    State: [cart_position, cart_velocity, pole_angle, pole_ang_velocity]
    Action: continuous force on cart
    """

    def __init__(self, gravity: float = 9.8, friction: float = 0.05, dt: float = 0.02):
        self.gravity = gravity
        self.friction = friction
        self.dt = dt
        self.state = np.zeros(4, dtype=np.float32)
        self.max_steps = 500
        self.steps = 0

    def reset(self, rng: np.random.RandomState | None = None):
        rng = rng or np.random.RandomState()
        self.state = rng.uniform(-0.1, 0.1, size=4).astype(np.float32)
        self.steps = 0
        return self.state.copy()

    def step(self, action: float):
        x, x_dot, theta, theta_dot = self.state
        force = np.clip(action, -10.0, 10.0)

        # Very simple dynamics (linearised-ish).
        x_ddot = (force - self.friction * x_dot) / 1.0
        theta_ddot = (self.gravity * np.sin(theta) - 0.5 * force * np.cos(theta)) / 0.5

        x_dot += x_ddot * self.dt
        x += x_dot * self.dt
        theta_dot += theta_ddot * self.dt
        theta += theta_dot * self.dt

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.steps += 1

        # Reward: keep pole near upright and cart near centre.
        reward = 1.0 - 0.5 * theta ** 2 - 0.01 * x ** 2 - 0.001 * (force ** 2)
        done = self.steps >= self.max_steps or abs(x) > 2.4 or abs(theta) > np.pi / 3
        return self.state.copy(), reward, done


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ASSTFPolicy(nn.Module):
    def __init__(self, state_dim: int = 4, action_dim: int = 1, rank: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            ASSTFBlock(state_dim, 32, structural_rank=rank, activation="relu"),
            ASSTFBlock(32, 16, structural_rank=rank, activation="relu"),
            nn.Linear(16, action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def build_static_baseline(state_dim: int = 4, action_dim: int = 1):
    return StaticMLP(state_dim, [48, 24], action_dim, activation="relu")


# ---------------------------------------------------------------------------
# Training (REINFORCE-like policy gradient)
# ---------------------------------------------------------------------------

def collect_episode(env, policy, device, rng):
    """Collect one episode and return (states, actions, rewards)."""
    states, actions, rewards = [], [], []
    state = env.reset(rng)
    done = False
    while not done:
        s_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            a_t = policy(s_t).cpu().item()
        next_state, reward, done = env.step(a_t * 5.0)  # scale action
        states.append(state)
        actions.append(a_t)
        rewards.append(reward)
        state = next_state
    return np.array(states), np.array(actions), np.array(rewards)


def compute_returns(rewards, gamma: float = 0.99):
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def train(
    n_episodes: int = 1000,
    asstf_episodes: int | None = None,
    gamma: float = 0.99,
    lr_core: float = 3e-4,
    lr_struct: float = 1e-4,
    static_patience: int = 50,
    asstf_patience: int | None = None,
    save_dir: Path | None = None,
):
    asstf_episodes = asstf_episodes or n_episodes
    asstf_patience = asstf_patience or static_patience * 10
    save_dir = save_dir or (ROOT / "checkpoints" / "app_05_rl")
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    asstf_policy = ASSTFPolicy(rank=3).to(device)
    static_policy = build_static_baseline().to(device)

    print("=" * 60)
    print("App 05: Online RL Dynamic Policy")
    print("=" * 60)
    print(f"ASSTF params:  {count_parameters(asstf_policy):,}")
    print(f"Static params: {count_parameters(static_policy):,}")

    asstf_trainer = BilevelTrainer(
        asstf_policy, lr_core=lr_core, lr_struct=lr_struct, device=device
    )
    static_opt = torch.optim.Adam(static_policy.parameters(), lr=lr_core)

    rng = np.random.RandomState(47)
    env = SimplePendulumEnv()

    asstf_rewards, static_rewards = [], []
    window = 50
    static_stopper = EarlyStopping(patience=static_patience, mode="max", min_delta=0.5)

    # Train static baseline until moving-average reward plateaus.
    for ep in range(n_episodes):
        states_s, actions_s, rewards_s = collect_episode(env, static_policy, device, rng)
        returns_s = compute_returns(rewards_s, gamma)
        static_rewards.append(rewards_s.sum())

        static_policy.train()
        s_t = torch.from_numpy(states_s).float().to(device)
        a_t = torch.from_numpy(actions_s).float().unsqueeze(-1).to(device)
        r_t = torch.from_numpy(returns_s).float().unsqueeze(-1).to(device)
        r_t = (r_t - r_t.mean()) / (r_t.std() + 1e-8)
        static_opt.zero_grad()
        log_prob = -((static_policy(s_t) - a_t) ** 2)
        loss = -(log_prob * r_t).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(static_policy.parameters(), 1.0)
        static_opt.step()

        if ep >= window - 1:
            avg_r = float(np.mean(static_rewards[-window:]))
            if (ep + 1) % 50 == 0:
                print(f"Episode {ep+1:03d}: Static reward={avg_r:.2f}")
            if static_stopper(avg_r):
                print(f"[Static] Early stopping at episode {ep+1} (best reward {static_stopper.best_value:.2f})")
                break

    asstf_stopper = EarlyStopping(patience=asstf_patience, mode="max", min_delta=0.5)
    # Train ASSTF until moving-average reward plateaus.
    for ep in range(asstf_episodes):
        states_a, actions_a, rewards_a = collect_episode(env, asstf_policy, device, rng)
        returns_a = compute_returns(rewards_a, gamma)
        asstf_rewards.append(rewards_a.sum())

        asstf_policy.train()
        s_t = torch.from_numpy(states_a).float().to(device)
        a_t = torch.from_numpy(actions_a).float().unsqueeze(-1).to(device)
        r_t = torch.from_numpy(returns_a).float().unsqueeze(-1).to(device)
        r_t = (r_t - r_t.mean()) / (r_t.std() + 1e-8)

        # Bilevel alternating update (Algorithm 1): core phase then structural phase.
        asstf_policy.train()

        def _pg_loss(logits):
            log_prob = -((logits - a_t) ** 2)
            return -(log_prob * r_t).mean()

        # Phase 1: update core parameters θc.
        asstf_trainer.optimizer_core.zero_grad()
        loss = _pg_loss(asstf_policy(s_t))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(asstf_policy.parameters(), 1.0)
        asstf_trainer.optimizer_core.step()

        # Phase 2: update structural parameters θs.
        asstf_trainer.optimizer_struct.zero_grad()
        loss = _pg_loss(asstf_policy(s_t))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(asstf_policy.parameters(), 1.0)
        asstf_trainer.optimizer_struct.step()

        if ep >= window - 1:
            avg_r = float(np.mean(asstf_rewards[-window:]))
            if (ep + 1) % 50 == 0:
                print(f"Episode {ep+1:03d}: ASSTF reward={avg_r:.2f}")
            if asstf_stopper(avg_r):
                print(f"[ASSTF] Early stopping at episode {ep+1} (best reward {asstf_stopper.best_value:.2f})")
                break

    torch.save(asstf_policy.state_dict(), save_dir / "asstf_model.pt")
    torch.save(static_policy.state_dict(), save_dir / "static_model.pt")

    metadata = {
        "app": "app_05_rl",
        "n_episodes": n_episodes,
        "asstf_episodes": asstf_episodes,
        "static_patience": static_patience,
        "asstf_patience": asstf_patience,
        "asstf_params": count_parameters(asstf_policy),
        "static_params": count_parameters(static_policy),
        "asstf_final_reward": float(np.mean(asstf_rewards[-50:])),
        "static_final_reward": float(np.mean(static_rewards[-50:])),
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved models to", save_dir)
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    train()
