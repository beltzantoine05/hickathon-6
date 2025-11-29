"""Model definitions for the DAE and finetuning head."""
from typing import Iterable, List

import torch
import torch.nn as nn


def _build_mlp(sizes: Iterable[int], activation: nn.Module, dropout: float | None = None) -> nn.Sequential:
    layers: List[nn.Module] = []
    sizes_list = list(sizes)
    for i in range(len(sizes_list) - 1):
        layers.append(nn.Linear(sizes_list[i], sizes_list[i + 1]))
        layers.append(type(activation)())
        if dropout is not None and dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class NoiseInjection(nn.Module):
    """Inject Gaussian noise scaled by observed standard deviation."""

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

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


class FlexibleEncoder(nn.Module):
    """Configurable encoder used by the denoising autoencoder."""

    def __init__(self, input_dim: int, hidden_sizes: List[int], dropout: float = 0.1):
        super().__init__()
        self.net = _build_mlp([input_dim, *hidden_sizes], activation=nn.GELU(), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlexibleDecoder(nn.Module):
    """Configurable decoder to reconstruct the masked input."""

    def __init__(self, latent_dim: int, hidden_sizes: List[int], output_dim: int):
        super().__init__()
        self.net = _build_mlp([latent_dim, *hidden_sizes, output_dim], activation=nn.GELU(), dropout=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DAE(nn.Module):
    """Denoising autoencoder with configurable architecture."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        encoder_hidden: List[int],
        decoder_hidden: List[int],
        alpha: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.noise_layer = NoiseInjection(alpha)
        self.encoder = FlexibleEncoder(input_dim=input_dim, hidden_sizes=encoder_hidden, dropout=dropout)
        latent_dim = encoder_hidden[-1] if encoder_hidden else input_dim
        self.decoder = FlexibleDecoder(latent_dim=latent_dim, hidden_sizes=decoder_hidden, output_dim=output_dim)

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


class MaskedSupervisedRegressor(nn.Module):
    """Regression head on top of a pretrained encoder."""

    def __init__(self, encoder: FlexibleEncoder, latent_dim: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, mask = x.chunk(2, dim=1)
        z = self.encoder(torch.cat([values * mask, mask], dim=1))
        return self.head(z).squeeze(-1)
