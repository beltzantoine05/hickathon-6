import torch
import torch.nn as nn


class NoiseInjection(nn.Module):
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


class Encoder(nn.Module):
    def __init__(self, input_dim: int = 220, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, output_dim: int = 110):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 220,
        output_dim: int = 110,
        alpha: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.noise_layer = NoiseInjection(alpha)
        self.encoder = Encoder(input_dim=input_dim, dropout=dropout)
        self.decoder = Decoder(output_dim=output_dim)

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


class SupervisedRegressor(nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)
