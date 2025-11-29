from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.nn as nn


def mlp(sizes: List[int], activation: nn.Module, dropout: float | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:  # no activation on last layer
            layers.append(activation)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class NoiseInjection(nn.Module):
    """Inject Gaussian noise scaled by feature-wise std and observed mask.

    Noise is applied only when `apply_noise` is True and `alpha > 0`.
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = float(alpha)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        std_vector_vals: torch.Tensor,
        apply_noise: bool,
    ) -> torch.Tensor:
        if not (apply_noise and self.alpha > 0):
            return values
        stdv = std_vector_vals.view(1, -1).to(device=values.device, dtype=values.dtype)
        noise = torch.randn_like(values)
        return values + (self.alpha * noise * stdv * mask)


class Encoder(nn.Module):
    """Configurable MLP encoder that maps concatenated [values || mask] to latent.

    Parameters
    ----------
    input_dim : int
        Dimension of concatenated input (n_features * 2).
    hidden : list[int]
        Hidden layer sizes.
    latent_dim : int
        Output latent dimension.
    dropout : float
        Dropout rate for hidden layers.
    """

    def __init__(self, input_dim: int, hidden: List[int], latent_dim: int, dropout: float = 0.1):
        super().__init__()
        sizes = [input_dim] + hidden + [latent_dim]
        self.net = mlp(sizes, nn.GELU(), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """Configurable MLP decoder that maps latent to reconstructed values.

    Parameters
    ----------
    output_dim : int
        Number of features to reconstruct.
    hidden : list[int]
        Hidden layer sizes (from latent to output).
    latent_dim : int
        Input latent dimension.
    """

    def __init__(self, output_dim: int, hidden: List[int], latent_dim: int):
        super().__init__()
        sizes = [latent_dim] + hidden + [output_dim]
        self.net = mlp(sizes, nn.GELU(), dropout=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DAE(nn.Module):
    """Denoising Autoencoder for masked tabular data.

    Concatenates imputed/scaled values with a binary mask of observed entries.
    Optionally injects noise on observed values during training.
    """

    def __init__(
        self,
        n_features: int,
        encoder_hidden: List[int],
        decoder_hidden: List[int],
        latent_dim: int = 64,
        alpha: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.output_dim = n_features
        self.noise_layer = NoiseInjection(alpha)
        self.encoder = Encoder(input_dim=2 * n_features, hidden=encoder_hidden, latent_dim=latent_dim, dropout=dropout)
        self.decoder = Decoder(output_dim=n_features, hidden=decoder_hidden, latent_dim=latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        std_vector: torch.Tensor | None = None,
        apply_noise: bool = False,
    ) -> torch.Tensor:
        values = x[:, : self.output_dim]
        mask = x[:, self.output_dim :]

        values = values * mask

        if std_vector is not None:
            std_vals = std_vector[: self.output_dim]
            values = self.noise_layer(values, mask, std_vals, apply_noise=apply_noise)

        x_in = torch.cat([values, mask], dim=1)
        latent = self.encoder(x_in)
        return self.decoder(latent)

    def encode(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_in = torch.cat([values * mask, mask], dim=1)
        return self.encoder(x_in)


class SupervisedRegressor(nn.Module):
    """Regression head on top of a frozen or trainable encoder.

    The head is a single linear layer by default mapping from latent_dim to 1.
    """

    def __init__(self, encoder: Encoder, latent_dim: int = 64):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)
