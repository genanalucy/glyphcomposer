from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None

from glyphgen.models.components import ConvBlock, Downsample, Upsample


class GlyphVAE(nn.Module):
    def __init__(self, image_channels: int = 1, latent_channels: int = 4, base_channels: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(image_channels, base_channels),
            Downsample(base_channels),
            ConvBlock(base_channels, base_channels * 2),
            Downsample(base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 4),
        )
        self.to_mu = nn.Conv2d(base_channels * 4, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv2d(base_channels * 4, latent_channels, kernel_size=1)
        self.from_latent = nn.Conv2d(latent_channels, base_channels * 4, kernel_size=1)
        self.decoder = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 4),
            Upsample(base_channels * 4),
            ConvBlock(base_channels * 4, base_channels * 2),
            Upsample(base_channels * 2),
            ConvBlock(base_channels * 2, base_channels),
            nn.Conv2d(base_channels, image_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def encode(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        hidden = self.encoder(x)
        return self.to_mu(hidden), self.to_logvar(hidden)

    def reparameterize(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return mu + noise * std

    def decode(self, z: "torch.Tensor") -> "torch.Tensor":
        return self.decoder(self.from_latent(z))

    def forward(self, x: "torch.Tensor") -> dict[str, "torch.Tensor"]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return {
            "reconstruction": reconstruction,
            "mu": mu,
            "logvar": logvar,
            "latents": z,
        }

