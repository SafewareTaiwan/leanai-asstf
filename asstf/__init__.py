"""
ASSTF (Adaptive State-Space Transfer Function) Reference Implementation
=======================================================================

This package provides a PyTorch reference implementation of the ASSTF
framework described in the LeanAI technical white-paper.  It is intended for
research, education, and non-commercial evaluation under the LeanAI ASSTF
Community License.  Commercial use requires a separate Commercial License.

Core modules
------------
core        : ASSTF layers and building blocks.
trainer     : Bilevel optimisation trainer (Algorithm 1 in the paper).
adaptation  : Inference-time structural adaptation via surprise minimisation.
"""

from .core import ASSTFLinear, ASSTFBlock, ASSTFConv1d, ASSTFConvBlock, count_parameters
from .trainer import BilevelTrainer
from .adaptation import SurpriseMinimizer

__all__ = [
    "ASSTFLinear",
    "ASSTFBlock",
    "ASSTFConv1d",
    "ASSTFConvBlock",
    "count_parameters",
    "BilevelTrainer",
    "SurpriseMinimizer",
]
