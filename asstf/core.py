"""
ASSTF core layers.

Reference mapping to the white-paper notation
---------------------------------------------
θc  : core parameters      -> self.weight, self.bias
θs  : structural parameters -> self.structural_u, self.structural_v,
                               self.structural_gate, self.alpha, self.zeta
Γ   : base transfer fn     -> F.linear(x, weight, bias)
Ψ   : structural modulator -> low-rank matrix M = U @ V, gated by sigmoid(gate)
⊛   : composition          -> element-wise addition: Γ + Ψ(x)
g_i : neuron gate          -> optional per-output input-activity gate
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LowRankProjectionMixin:
    """
    Shared helpers for soft-thresholding a structural modulation matrix.

    Both ``ASSTFLinear`` and ``ASSTFConv1d`` factorise their structural
    branch as ``U @ V`` and optionally project it toward low-rank structure.
    This mixin provides the projection backends so that the ``svd_method``
    argument is honoured consistently across layer types.
    """

    def low_rank_projection(self, M: torch.Tensor, method: str = "elementwise") -> torch.Tensor:
        """
        Apply a soft-thresholding projection to encourage low-rank structure.

        Parameters
        ----------
        method : {"elementwise", "full_svd", "randomized_svd", "power_iter"}
            - elementwise: cheap entry-wise soft thresholding.  Not a true
              singular-value threshold but acts as a sparsifying regulariser
              and is fully differentiable without any SVD.
            - full_svd: exact singular-value soft thresholding (Eq. 4).
            - randomized_svd: approximate SVD via random projections; faster
              for large matrices.
            - power_iter: single-vector power iteration to approximate the
              top singular value and shrink the whole matrix accordingly.
        """
        if method == "elementwise":
            return self._elementwise_soft_threshold(M)
        if method == "full_svd":
            return self._svd_soft_threshold(M)
        if method == "randomized_svd":
            return self._randomized_svd_soft_threshold(M)
        if method == "power_iter":
            return self._power_iter_soft_threshold(M)
        raise ValueError(f"Unknown svd_method: {method}")

    def _elementwise_soft_threshold(self, M: torch.Tensor) -> torch.Tensor:
        """
        Differentiable, SVD-free soft thresholding.

        Shrink each entry toward zero based on an adaptive threshold.  This
        is much cheaper than SVD and empirically encourages sparse / low-rank
        structural matrices.
        """
        threshold = torch.clamp(self.alpha, min=1e-6) * M.abs().mean()
        beta = torch.clamp(self.beta, min=1e-2)
        shrink = torch.sigmoid(beta * (M.abs() - threshold))
        return M * shrink

    def _svd_soft_threshold(self, M: torch.Tensor) -> torch.Tensor:
        """Exact singular-value soft thresholding (Eq. 4)."""
        if not torch.isfinite(M).all():
            return M
        try:
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        except RuntimeError:
            return M
        if not torch.isfinite(S).all():
            return M
        threshold = torch.clamp(self.alpha, min=1e-6) * S.sum()
        beta = torch.clamp(self.beta, min=1e-2)
        shrink = torch.sigmoid(beta * (S - threshold))
        S_tilde = S * shrink
        return U @ torch.diag_embed(S_tilde) @ Vh

    def _randomized_svd_soft_threshold(
        self, M: torch.Tensor, n_oversamples: int = 2, n_iter: int = 1
    ) -> torch.Tensor:
        """
        Approximate singular-value soft thresholding via randomized SVD.

        Much faster than full SVD for wide/tall matrices.  The target rank is
        the structural rank plus a small number of oversamples.
        """
        if not torch.isfinite(M).all():
            return M

        k = min(self.structural_rank + n_oversamples, min(M.shape))
        # Gaussian random projection matrix.
        Omega = torch.randn(M.size(1), k, device=M.device, dtype=M.dtype)
        Y = M @ Omega  # (out, k)

        # Power iteration to improve approximation.
        for _ in range(n_iter):
            Y = M @ (M.T @ Y)

        try:
            Q, _ = torch.linalg.qr(Y)
            B = Q.T @ M  # (k, in)
            U_hat, S, Vh = torch.linalg.svd(B, full_matrices=False)
        except RuntimeError:
            return M

        if not torch.isfinite(S).all():
            return M

        U = Q @ U_hat
        threshold = torch.clamp(self.alpha, min=1e-6) * S.sum()
        beta = torch.clamp(self.beta, min=1e-2)
        shrink = torch.sigmoid(beta * (S - threshold))
        S_tilde = S * shrink
        return U @ torch.diag_embed(S_tilde) @ Vh

    def _power_iter_soft_threshold(self, M: torch.Tensor, n_iter: int = 2) -> torch.Tensor:
        """
        Approximate top-singular-value soft thresholding using power iteration.

        Cheapest of the SVD-based options; scales linearly with matrix size.
        """
        if not torch.isfinite(M).all():
            return M

        # Start with a random vector.
        v = torch.randn(M.size(1), 1, device=M.device, dtype=M.dtype)
        v = v / (v.norm() + 1e-8)

        for _ in range(n_iter):
            v = M.T @ (M @ v)
            v = v / (v.norm() + 1e-8)

        sigma = (M @ v).norm()
        if not torch.isfinite(sigma):
            return M

        threshold = torch.clamp(self.alpha, min=1e-6) * sigma
        beta = torch.clamp(self.beta, min=1e-2)
        shrink = torch.sigmoid(beta * (M.abs() - threshold))
        return M * shrink


class ASSTFLinear(nn.Module, _LowRankProjectionMixin):
    """
    Drop-in replacement for a standard ``nn.Linear`` layer using the ASSTF
    framework.

    The forward pass is:

        base = x @ W_c^T + b_c                       (Γ, core transform)
        M    = sigmoid(gate) * (U @ V)               (Ψ, low-rank structural modulator)
        out  = base + x @ M^T                        (Γ ⊛ Ψ)

    The structural branch is intentionally low-rank so that the effective
    state-space dimension of the neuron can be adapted continuously during
    training and inference.  The scalar ``structural_gate`` acts as a
    differentiable soft-threshold, suppressing the structural branch when its
    contribution is not needed (analogous to the soft singular-value
    thresholding discussed in Eq. (4) of the paper).

    Parameters
    ----------
    in_features : int
    out_features : int
    structural_rank : int, optional
        Rank of the low-rank structural modulation matrix (default: 4).
    use_neuron_gate : bool, optional
        If True, multiply the output by a per-output sigmoid gate that depends
        on the mean absolute activity of the inputs.  This is a practical
        realisation of the gating function ``g_i(t)`` in Eq. (8).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        structural_rank: int = 4,
        use_neuron_gate: bool = False,
        use_svd_projection: bool = False,
        svd_method: str = "elementwise",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.structural_rank = structural_rank
        self.use_neuron_gate = use_neuron_gate
        self.use_svd_projection = use_svd_projection
        self.svd_method = svd_method

        # Core parameters θc
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        # Structural parameters θs
        self.structural_u = nn.Parameter(torch.empty(out_features, structural_rank))
        self.structural_v = nn.Parameter(torch.empty(structural_rank, in_features))
        # Initialize gate to a strongly negative value so sigmoid(gate) ≈ 0.
        # This makes the structural branch inactive at start, so the layer
        # initially behaves exactly like a standard Linear layer.
        self.structural_gate = nn.Parameter(torch.tensor(-6.0))

        # Adaptive soft-thresholding hyperparameters (Eq. 4)
        # These are kept as small learnable parameters so the network can adapt
        # the thresholding aggressiveness.  They correspond to α, ζ and β in
        # the paper's notation.
        self.alpha = nn.Parameter(torch.tensor(0.05))
        self.zeta = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(8.0))

        # Optional per-output activity gate (Eq. 8)
        if use_neuron_gate:
            self.gate_scale = nn.Parameter(torch.ones(out_features))
            self.gate_bias = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Standard Kaiming init for the core transform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # Structural branch starts near zero so the layer initially behaves
        # like a standard linear layer.
        nn.init.xavier_uniform_(self.structural_u)
        nn.init.xavier_uniform_(self.structural_v)
        # Keep gate strongly negative at reset so the structural branch starts
        # near zero and the layer behaves like a plain Linear layer.
        nn.init.constant_(self.structural_gate, -6.0)

    def structural_modulator(self, grad_struct: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Build the structural modulation matrix Ψ.

        If ``grad_struct`` is supplied, its Frobenius norm is used to damp the
        structural contribution according to Eq. (3) in the paper.
        """
        M = self.structural_u @ self.structural_v  # (out, in)

        # Differentiable soft-thresholding gate.
        # The gate suppresses the structural branch when it is not useful.
        gate = torch.sigmoid(self.structural_gate)

        # Optional damping inspired by Eq. (3): exp(-ζ * ||∇θs L||_F)
        if grad_struct is not None and grad_struct.numel() > 0:
            damping = torch.exp(-torch.clamp(self.zeta, min=1e-6) * grad_struct.norm())
        else:
            damping = torch.tensor(1.0, device=M.device, dtype=M.dtype)

        return gate * damping * M

    def neuron_gate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input-dependent per-output gate g_i(t) (Eq. 8 simplified).

        Measures average input activity and uses it to modulate each output
        neuron independently.
        """
        if not self.use_neuron_gate:
            return torch.ones(self.out_features, device=x.device, dtype=x.dtype)

        # Mean absolute activity over the feature dimension.
        activity = x.abs().mean(dim=-1, keepdim=True)  # (B, 1)
        logits = self.gate_scale * activity + self.gate_bias  # (B, out)
        return torch.sigmoid(logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing φ(t; x, Θ) = Γ(x, θc) ⊛ Ψ(∇ΘL, θs).

        The structural gradient ``grad_struct`` is not passed during ordinary
        forward propagation; it can be supplied by the trainer for the damping
        term if desired.
        """
        # Γ(x, θc)
        base = F.linear(x, self.weight, self.bias)

        # Ψ(∇ΘL, θs)
        M = self.structural_modulator(grad_struct=None)
        # Optional soft-rank projection (Eq. 4).  Disabled by default because
        # full SVD on every forward pass can be expensive and numerically
        # unstable in mixed precision.  The differentiable gate already
        # provides a soft-thresholding effect.  When enabled, users can choose
        # a more efficient approximation method.
        if self.training and self.use_svd_projection:
            M = self.low_rank_projection(M, method=self.svd_method)
        struct = F.linear(x, M, None)

        # Composition ⊛: additive composition for stable gradients
        out = base + struct

        # Optional neuron-level activity gate
        g = self.neuron_gate(x)
        out = out * g

        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"structural_rank={self.structural_rank}, "
            f"use_neuron_gate={self.use_neuron_gate}, "
            f"use_svd_projection={self.use_svd_projection}, "
            f"svd_method={self.svd_method}"
        )


class ASSTFBlock(nn.Module):
    """
    Convenience block: ASSTF linear layer + optional activation + optional
    batch/layer normalisation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        structural_rank: int = 4,
        activation: Optional[str] = "relu",
        use_neuron_gate: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear = ASSTFLinear(
            in_features,
            out_features,
            structural_rank=structural_rank,
            use_neuron_gate=use_neuron_gate,
        )

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation is None:
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.linear(x)))


class ASSTFConv1d(nn.Module, _LowRankProjectionMixin):
    """
    Drop-in ASSTF replacement for ``nn.Conv1d``.

    The design mirrors ``ASSTFLinear`` and the ASSTF-YOLO reference:
    a core convolution carries the main signal, while a low-rank
    structural convolution is added as a residual adapter.

    The structural kernel is factorised as ``U @ V`` and reshaped to
    ``(out_channels, in_channels, kernel_size)`` so that the parameter
    overhead is only ``rank * (out_channels + in_channels * kernel_size)``
    instead of a full additional convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        structural_rank: int = 4,
        use_svd_projection: bool = False,
        svd_method: str = "elementwise",
    ) -> None:
        super().__init__()
        if groups != 1:
            raise ValueError("ASSTFConv1d currently only supports groups=1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.structural_rank = structural_rank
        self.use_svd_projection = use_svd_projection
        self.svd_method = svd_method

        # Core convolution θc
        self.core = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # Structural low-rank factors θs
        self.structural_u = nn.Parameter(torch.empty(out_channels, structural_rank))
        self.structural_v = nn.Parameter(torch.empty(structural_rank, in_channels * kernel_size))
        # Strongly negative gate so the structural branch is off at start.
        self.structural_gate = nn.Parameter(torch.tensor(-6.0))

        # Adaptive soft-thresholding hyperparameters
        self.alpha = nn.Parameter(torch.tensor(0.05))
        self.zeta = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(8.0))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Core conv keeps PyTorch's default init.
        nn.init.kaiming_uniform_(self.core.weight, a=math.sqrt(5))
        if self.core.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.core.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.core.bias, -bound, bound)

        # Structural branch starts near zero.
        nn.init.xavier_uniform_(self.structural_u)
        nn.init.xavier_uniform_(self.structural_v)
        nn.init.constant_(self.structural_gate, -6.0)

    def structural_kernel(self) -> torch.Tensor:
        """Build and optionally project the structural convolution kernel."""
        M = self.structural_u @ self.structural_v  # (out, in*k)

        if self.training and self.use_svd_projection:
            M = self._project_structural(M)

        return M.view(self.out_channels, self.in_channels, self.kernel_size)

    def _project_structural(self, M: torch.Tensor) -> torch.Tensor:
        """Apply the selected low-rank projection to the flattened kernel."""
        return self.low_rank_projection(M, method=self.svd_method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.core(x)

        struct_k = self.structural_kernel()
        gate = torch.sigmoid(self.structural_gate)
        struct = gate * F.conv1d(
            x,
            struct_k,
            None,
            stride=self.core.stride,
            padding=self.core.padding,
            dilation=self.core.dilation,
            groups=self.core.groups,
        )

        return base + struct

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.core.stride}, padding={self.core.padding}, "
            f"structural_rank={self.structural_rank}"
        )
        return s


class ASSTFConvBlock(nn.Module):
    """Convenience block: ASSTF 1-D convolution + activation + optional BN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        structural_rank: int = 4,
        activation: Optional[str] = "relu",
        batch_norm: bool = False,
        **conv_kwargs,
    ) -> None:
        super().__init__()
        self.conv = ASSTFConv1d(
            in_channels,
            out_channels,
            kernel_size,
            structural_rank=structural_rank,
            **conv_kwargs,
        )
        self.bn = nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity()

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation is None:
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return the number of (trainable) parameters in a model."""
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))
