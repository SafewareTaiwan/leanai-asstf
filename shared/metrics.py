"""
Common metrics used across the 5+1 ASSTF applications.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Classification accuracy.

    ``logits`` may be ``(B, C)`` or ``(B,)`` for binary single-logit cases.
    ``labels`` may be class indices ``(B,)``, one-hot / soft labels of the same
    shape as ``logits``, or ``(B, 1)``.  The function always compares
    ``preds == labels`` after flattening both tensors to the same length, so
    no broadcasting occurs.
    """
    preds = torch.argmax(logits, dim=-1)
    labels = labels.to(logits.device)

    # Convert one-hot / soft labels to class indices only when the shapes
    # fully match and the last dimension is a class dimension.
    if labels.dim() == logits.dim() and labels.shape == logits.shape and labels.size(-1) > 1:
        labels = torch.argmax(labels, dim=-1)
    elif labels.dim() > 1:
        labels = labels.squeeze(-1)

    if preds.dim() != 1:
        preds = preds.view(-1)
    if labels.dim() != 1:
        labels = labels.view(-1)

    if preds.numel() != labels.numel():
        raise ValueError(
            f"preds and labels must have the same number of elements, "
            f"got {preds.shape} and {labels.shape}"
        )

    correct = (preds == labels).float().sum().item()
    return correct / labels.numel()


def f1_score(logits: torch.Tensor, labels: torch.Tensor, num_classes: int | None = None) -> float:
    """
    Macro-averaged F1 score.

    For binary classification logits can be either (B, 1) or (B, 2).
    """
    preds = torch.argmax(logits, dim=-1)
    if labels.dim() == logits.dim() and labels.size(-1) > 1:
        labels = torch.argmax(labels, dim=-1)
    elif labels.dim() > 1:
        labels = labels.squeeze(-1)

    labels = labels.long()
    num_classes = num_classes or int(max(preds.max().item(), labels.max().item()) + 1)

    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    return sum(f1s) / len(f1s)


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean squared error."""
    return F.mse_loss(pred, target).item()


def reconstruction_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Alias for MSE, commonly used in anomaly detection."""
    return mse(pred, target)


def auc_roc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Approximate ROC-AUC using the Mann-Whitney U statistic.

    ``scores`` should be anomaly scores (higher = more anomalous).
    ``labels`` should be 0/1 where 1 = anomaly.
    """
    scores = scores.view(-1)
    labels = labels.view(-1).bool()

    pos_scores = scores[labels]
    neg_scores = scores[~labels]

    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return 0.5

    # Mann-Whitney U
    n_pos = pos_scores.numel()
    n_neg = neg_scores.numel()

    pos_scores = pos_scores.sort().values
    neg_scores = neg_scores.sort().values

    # Compute rank of positive samples in combined sorted array.
    combined = torch.cat([pos_scores, neg_scores]).sort().values
    ranks = []
    for s in pos_scores:
        # average rank for ties
        idx = (combined == s).nonzero(as_tuple=True)[0]
        rank = (idx.min().item() + idx.max().item()) / 2.0 + 1
        ranks.append(rank)

    rank_sum = sum(ranks)
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)
