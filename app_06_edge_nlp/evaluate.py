"""
App 06: Edge NLP Intent Classification - Evaluation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app_06_edge_nlp.train import TinyBertAllASSTF96, TinyBertFFN, load_data
from asstf import count_parameters


def evaluate(
    checkpoint_dir: Path | None = None,
    output_dir: Path | None = None,
):
    checkpoint_dir = checkpoint_dir or (ROOT / "checkpoints" / "app_06_edge_nlp")
    output_dir = output_dir or (ROOT / "results" / "app_06_edge_nlp")
    output_dir.mkdir(parents=True, exist_ok=True)

    _, test_loader, vocab_size = load_data(batch_size=32)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    asstf_model = TinyBertAllASSTF96(
        vocab_size, hidden_size=96, num_layers=2, rank=4
    ).to(device)
    asstf_model.load_state_dict(
        torch.load(checkpoint_dir / "asstf_model.pt", map_location=device)
    )

    static_model = TinyBertFFN(vocab_size, hidden_size=128, num_layers=2).to(device)
    static_model.load_state_dict(
        torch.load(checkpoint_dir / "static_model.pt", map_location=device)
    )

    def _accuracy(model, loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == y).sum().item()
                total += x.size(0)
        return correct / total

    asstf_acc = _accuracy(asstf_model, test_loader)
    static_acc = _accuracy(static_model, test_loader)

    result = {
        "app": "app_06_edge_nlp",
        "data": "real_sst2_sentencepiece",
        "asstf_params": count_parameters(asstf_model),
        "static_params": count_parameters(static_model),
        "asstf_test_acc": asstf_acc,
        "static_test_acc": static_acc,
        "param_reduction_pct": (
            1.0 - count_parameters(asstf_model) / count_parameters(static_model)
        ) * 100.0,
    }
    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nFinal:", json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    evaluate()
