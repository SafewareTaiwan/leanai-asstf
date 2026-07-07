"""
Simple static baselines used for comparison against ASSTF models.

Each baseline is intentionally built with roughly the same layer topology as
its ASSTF counterpart, but using standard PyTorch layers.  Parameter counts
are typically larger because static models rely on full-rank transformations
instead of ASSTF's adaptive low-rank structural branch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticMLP(nn.Module):
    """Plain MLP baseline."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        act = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}.get(
            activation, nn.ReLU
        )

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StaticCNN1D(nn.Module):
    """Plain 1D CNN baseline for sequence / audio tasks."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        seq_len: int,
        channels: list = [16, 32],
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        conv_layers = []
        prev = in_channels
        for ch in channels:
            conv_layers.append(nn.Conv1d(prev, ch, kernel_size, padding=kernel_size // 2))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(2))
            if dropout > 0:
                conv_layers.append(nn.Dropout(dropout))
            prev = ch
        self.conv = nn.Sequential(*conv_layers)

        # Compute flattened size after convolutions.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, seq_len)
            feat = self.conv(dummy).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(feat, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        feats = self.conv(x)
        feats = feats.view(feats.size(0), -1)
        return self.classifier(feats)


class StaticAutoencoder(nn.Module):
    """Plain LSTM / MLP autoencoder baseline for anomaly detection."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 64,
        use_lstm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_lstm = use_lstm

        if use_lstm:
            self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.latent = nn.Linear(hidden_dim, latent_dim)
            self.decoder_input = nn.Linear(latent_dim, hidden_dim)
            self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_lstm:
            out, _ = self.encoder(x)  # (B, T, H)
            return self.latent(out[:, -1, :])  # (B, latent)
        return self.encoder(x)

    def decode(self, z: torch.Tensor, seq_len: int | None = None) -> torch.Tensor:
        if self.use_lstm:
            if seq_len is None:
                raise ValueError("seq_len is required for LSTM decoder")
            h = self.decoder_input(z).unsqueeze(1).repeat(1, seq_len, 1)
            out, _ = self.decoder(h)
            return out
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_lstm:
            z = self.encode(x)
            recon = self.decode(z, seq_len=x.size(1))
        else:
            z = self.encode(x)
            recon = self.decode(z)
        return recon
