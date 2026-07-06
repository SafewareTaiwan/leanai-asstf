"""
Simple early-stopping helper for training until a validation metric plateaus.
"""

from __future__ import annotations

import math


class EarlyStopping:
    """
    Stop training when a monitored metric has not improved for ``patience``
    checks.

    Parameters
    ----------
    patience : int
        Number of checks with no improvement after which training stops.
    mode : {"min", "max"}
        Whether the metric should be minimized or maximized.
    min_delta : float
        Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 10, mode: str = "min", min_delta: float = 1e-6):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = math.inf if mode == "min" else -math.inf
        self.early_stop = False

    def __call__(self, value: float) -> bool:
        """Return True if training should stop."""
        improved = False
        if self.mode == "min":
            if value < self.best_value - self.min_delta:
                self.best_value = value
                improved = True
        else:
            if value > self.best_value + self.min_delta:
                self.best_value = value
                improved = True

        if improved:
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self):
        self.counter = 0
        self.best_value = math.inf if self.mode == "min" else -math.inf
        self.early_stop = False
