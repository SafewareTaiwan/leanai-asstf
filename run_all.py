"""
Run all 5+1 ASSTF application demos and aggregate benchmarks.

Usage:
    python run_all.py [--quick]

The ``--quick`` flag reduces training epochs/iterations for a fast smoke run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app_01_gesture.train import train as train_gesture
from app_01_gesture.evaluate import evaluate as eval_gesture
from app_02_wake_word.train import train as train_wake
from app_02_wake_word.evaluate import evaluate as eval_wake
from app_03_anomaly.train import train as train_anomaly
from app_03_anomaly.evaluate import evaluate as eval_anomaly
from app_04_few_shot.train import train as train_few_shot
from app_04_few_shot.evaluate import evaluate as eval_few_shot
from app_05_rl.train import train as train_rl
from app_05_rl.evaluate import evaluate as eval_rl
from app_06_edge_nlp.train import train as train_nlp
from app_06_edge_nlp.evaluate import evaluate as eval_nlp


APP_REGISTRY = [
    ("app_01_gesture", train_gesture, eval_gesture),
    ("app_02_wake_word", train_wake, eval_wake),
    ("app_03_anomaly", train_anomaly, eval_anomaly),
    ("app_04_few_shot", train_few_shot, eval_few_shot),
    ("app_05_rl", train_rl, eval_rl),
    ("app_06_edge_nlp", train_nlp, eval_nlp),
]


def main():
    parser = argparse.ArgumentParser(description="Run all ASSTF 5+1 demos")
    parser.add_argument("--quick", action="store_true", help="Run a quick smoke test")
    args = parser.parse_args()

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for app_name, train_fn, eval_fn in APP_REGISTRY:
        print("\n" + "=" * 70)
        print(f"Running {app_name}")
        print("=" * 70)
        start = time.time()

        try:
            if args.quick:
                # Reduce workload for quick smoke test.
                if app_name == "app_01_gesture":
                    train_fn(epochs=3)
                elif app_name == "app_02_wake_word":
                    train_fn(epochs=3)
                elif app_name == "app_03_anomaly":
                    train_fn(epochs=3)
                elif app_name == "app_04_few_shot":
                    train_fn(n_tasks=50)
                elif app_name == "app_05_rl":
                    train_fn(n_episodes=30)
                elif app_name == "app_06_edge_nlp":
                    train_fn(epochs=1, asstf_epochs=1, max_samples=500)
            else:
                train_fn()

            eval_result = eval_fn()
            elapsed = time.time() - start
            summary = {
                "app": app_name,
                "status": "ok",
                "elapsed_sec": round(elapsed, 2),
                "result": eval_result,
            }
        except Exception as exc:
            elapsed = time.time() - start
            summary = {
                "app": app_name,
                "status": "error",
                "elapsed_sec": round(elapsed, 2),
                "error": str(exc),
            }
            print(f"[ERROR] {app_name}: {exc}")

        summaries.append(summary)

    benchmark_path = results_dir / "benchmarks.json"
    with open(benchmark_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)
    print(f"Benchmarks written to {benchmark_path}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
