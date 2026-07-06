"""
Unit tests for the ASSTF core implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from asstf import ASSTFLinear, ASSTFBlock, ASSTFConv1d, BilevelTrainer, SurpriseMinimizer, count_parameters


def test_asstf_linear_output_shape():
    layer = ASSTFLinear(10, 5, structural_rank=2)
    x = torch.randn(4, 10)
    y = layer(x)
    assert y.shape == (4, 5)


def test_asstf_block_output_shape():
    block = ASSTFBlock(10, 5, structural_rank=2, activation="relu")
    x = torch.randn(4, 10)
    y = block(x)
    assert y.shape == (4, 5)


def test_asstf_linear_gradients_flow():
    layer = ASSTFLinear(10, 5, structural_rank=2)
    x = torch.randn(4, 10, requires_grad=True)
    y = layer(x).sum()
    y.backward()
    assert layer.weight.grad is not None
    assert layer.structural_u.grad is not None
    assert layer.structural_v.grad is not None


def test_bilevel_trainer_split():
    model = nn.Sequential(
        ASSTFLinear(8, 8, structural_rank=2),
        ASSTFLinear(8, 2, structural_rank=2),
    )
    trainer = BilevelTrainer(model, lr_core=1e-3, lr_struct=1e-3)
    assert len(trainer.core_params) > 0
    assert len(trainer.struct_params) > 0


def test_surprise_minimizer_adapts():
    model = nn.Sequential(
        ASSTFLinear(8, 8, structural_rank=2),
        ASSTFLinear(8, 8, structural_rank=2),
    )
    x = torch.randn(4, 8)
    adapter = SurpriseMinimizer(model, lr=1e-2, max_steps=10)
    u_before = model[0].structural_u.detach().clone()
    loss = adapter.adapt(x)
    assert isinstance(loss, float)
    # Adaptation must actually change a structural parameter.
    assert not torch.allclose(model[0].structural_u, u_before)


def test_count_parameters():
    model = ASSTFLinear(10, 5, structural_rank=2)
    n = count_parameters(model)
    assert n > 0


def test_asstf_conv1d_output_shape():
    layer = ASSTFConv1d(1, 4, kernel_size=5, padding=2, structural_rank=1)
    x = torch.randn(2, 1, 64)
    y = layer(x)
    assert y.shape == (2, 4, 64)


def test_asstf_conv1d_gradients_flow():
    layer = ASSTFConv1d(2, 3, kernel_size=3, padding=1, structural_rank=1)
    x = torch.randn(2, 2, 16, requires_grad=True)
    y = layer(x).sum()
    y.backward()
    assert layer.core.weight.grad is not None
    assert layer.structural_u.grad is not None
    assert layer.structural_v.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
