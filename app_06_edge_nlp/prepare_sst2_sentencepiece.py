"""
Prepare real SST-2 with a small custom SentencePiece tokenizer.

This keeps the vocabulary at 3000 (matching the synthetic fallback) so that
embedding parameters do not explode when switching from synthetic to real data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(
    max_samples: int = 4000,
    max_length: int = 64,
    vocab_size: int = 3000,
):
    from datasets import load_dataset
    import sentencepiece as spm

    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("[Data] Loading SST-2 from HuggingFace...")
    ds = load_dataset("nyu-mll/glue", "sst2", split="train")
    texts = ds["sentence"][:max_samples]
    labels = ds["label"][:max_samples]

    raw_path = data_dir / "sst2_raw.txt"
    with open(raw_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")
    print(f"[Data] Wrote {len(texts)} lines to {raw_path}")

    model_prefix = data_dir / "sst2_spm"
    print(f"[Tokenizer] Training SentencePiece (vocab={vocab_size})...")
    spm.SentencePieceTrainer.train(
        input=str(raw_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        num_threads=4,
    )

    sp = spm.SentencePieceProcessor()
    sp.load(str(model_prefix) + ".model")
    pad_id = sp.pad_id()

    X = np.full((len(texts), max_length), pad_id, dtype=np.int64)
    for i, t in enumerate(texts):
        ids = sp.encode_as_ids(t)
        if len(ids) > max_length:
            ids = ids[:max_length]
        X[i, : len(ids)] = ids

    y = np.array(labels, dtype=np.int64)

    out_path = data_dir / "sst2_sentencepiece.npz"
    np.savez(out_path, X=X, y=y, vocab_size=vocab_size)
    print(f"[Data] Saved encoded data to {out_path}")
    print(f"[Data] X shape: {X.shape}, y shape: {y.shape}, vocab_size: {vocab_size}")


if __name__ == "__main__":
    main()
