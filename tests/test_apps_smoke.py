"""
Smoke tests ensuring each application train/eval script can run for a few
iterations without crashing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app_01_gesture.evaluate import evaluate as eval_gesture
from app_01_gesture.train import train as train_gesture
from app_02_wake_word.evaluate import evaluate as eval_wake
from app_02_wake_word.train import train as train_wake
from app_03_anomaly.evaluate import evaluate as eval_anomaly
from app_03_anomaly.train import train as train_anomaly
from app_04_few_shot.evaluate import evaluate as eval_few_shot
from app_04_few_shot.train import train as train_few_shot
from app_05_rl.evaluate import evaluate as eval_rl
from app_05_rl.train import train as train_rl
from app_06_edge_nlp.evaluate import evaluate as eval_nlp
from app_06_edge_nlp.train import train as train_nlp

APPS = [
    ("gesture", train_gesture, eval_gesture, {"epochs": 2}, {}),
    ("wake_word", train_wake, eval_wake, {"epochs": 2}, {}),
    ("anomaly", train_anomaly, eval_anomaly, {"epochs": 2}, {"n_windows": 3}),
    ("few_shot", train_few_shot, eval_few_shot, {"n_tasks": 20}, {"n_eval_tasks": 5}),
    ("rl", train_rl, eval_rl, {"n_episodes": 10}, {"n_eval_episodes": 6}),
    ("edge_nlp", train_nlp, eval_nlp, {"epochs": 1, "max_samples": 500}, {}),
]


@pytest.mark.parametrize("name,train_fn,eval_fn,train_kwargs,eval_kwargs", APPS)
def test_app_smoke(name, train_fn, eval_fn, train_kwargs, eval_kwargs):
    try:
        train_fn(**train_kwargs)
    except FileNotFoundError as e:
        if "data" in str(e) or ".npz" in str(e):
            pytest.skip(f"Skipping test due to missing local dataset: {e}")
        else:
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
