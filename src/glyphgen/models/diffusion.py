from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None
    F = None

from glyphgen.models.components import (
    ConditionEncoder,
    ConvBlock,
    Downsample,
    SinusoidalTimeEmbedding,
    Upsample,
    broadcast_condition,
)
from glyphgen.structures import STRUCTURE_TYPES


class LatentDenoiser(nn.Module):
    def __init__(self, latent_channels: int, condition_dim: int, base_channels: int = 128) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(condition_dim),
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        in_channels = latent_channels + 2 + condition_dim
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.down1 = Downsample(base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.down2 = Downsample(base_channels * 2)
        self.mid = ConvBlock(base_channels * 2, base_channels * 4)
        self.up2 = Upsample(base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.up1 = Upsample(base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 3, base_channels)
        self.out = nn.Conv2d(base_channels, latent_channels, kernel_size=1)

    def forward(
        self,
        latents: "torch.Tensor",
        layout_heatmap: "torch.Tensor",
        condition: "torch.Tensor",
        timesteps: "torch.Tensor",
    ) -> "torch.Tensor":
        time_condition = condition + self.time_embed(timesteps)
        layout_resized = F.interpolate(layout_heatmap, size=latents.shape[-2:], mode="bilinear", align_corners=False)
        condition_map = broadcast_condition(time_condition, latents.shape[-2:])
        x = torch.cat([latents, layout_resized, condition_map], dim=1)
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.down1(skip1))
        mid = self.mid(self.down2(skip2))
        x = self.up2(mid)
        x = self.dec2(torch.cat([x, skip2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, skip1], dim=1))
        return self.out(x)


class GlyphLatentDiffusion(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        style_vocab_size: int,
        latent_channels: int = 4,
        timesteps: int = 200,
        condition_embedding_dim: int = 128,
        base_channels: int = 128,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.condition_encoder = ConditionEncoder(
            vocab_size=vocab_size,
            style_vocab_size=style_vocab_size,
            structure_vocab_size=len(STRUCTURE_TYPES),
            embedding_dim=condition_embedding_dim,
        )
        self.denoiser = LatentDenoiser(
            latent_channels=latent_channels,
            condition_dim=condition_embedding_dim,
            base_channels=base_channels,
        )
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def encode_condition(
        self,
        *,
        component_images: "torch.Tensor",
        component_ids: "torch.Tensor",
        style_index: "torch.Tensor",
        structure_index: "torch.Tensor",
    ) -> "torch.Tensor":
        return self.condition_encoder(
            component_images=component_images,
            component_ids=component_ids,
            style_index=style_index,
            structure_index=structure_index,
        )

    def q_sample(self, x_start: "torch.Tensor", t: "torch.Tensor", noise: "torch.Tensor") -> "torch.Tensor":
        return (
            self.sqrt_alphas_cumprod[t][:, None, None, None] * x_start
            + self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * noise
        )

    def training_loss(
        self,
        *,
        latents: "torch.Tensor",
        component_images: "torch.Tensor",
        component_ids: "torch.Tensor",
        style_index: "torch.Tensor",
        structure_index: "torch.Tensor",
        layout_heatmap: "torch.Tensor",
    ) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
        batch_size = latents.shape[0]
        timesteps = torch.randint(0, self.timesteps, (batch_size,), device=latents.device)
        noise = torch.randn_like(latents)
        noisy_latents = self.q_sample(latents, timesteps, noise)
        condition = self.encode_condition(
            component_images=component_images,
            component_ids=component_ids,
            style_index=style_index,
            structure_index=structure_index,
        )
        predicted_noise = self.denoiser(noisy_latents, layout_heatmap, condition, timesteps)
        loss = F.mse_loss(predicted_noise, noise)
        return loss, {"timesteps": timesteps, "predicted_noise": predicted_noise}

    @torch.no_grad()
    def sample(
        self,
        *,
        latent_shape: tuple[int, int, int, int],
        component_images: "torch.Tensor",
        component_ids: "torch.Tensor",
        style_index: "torch.Tensor",
        structure_index: "torch.Tensor",
        layout_heatmap: "torch.Tensor",
    ) -> "torch.Tensor":
        latents = torch.randn(latent_shape, device=component_images.device)
        condition = self.encode_condition(
            component_images=component_images,
            component_ids=component_ids,
            style_index=style_index,
            structure_index=structure_index,
        )

        for step in reversed(range(self.timesteps)):
            t = torch.full((latent_shape[0],), step, device=latents.device, dtype=torch.long)
            noise_pred = self.denoiser(latents, layout_heatmap, condition, t)
            alpha = self.alphas[step]
            alpha_bar = self.alphas_cumprod[step]
            beta = self.betas[step]
            latents = (latents - beta / torch.sqrt(1.0 - alpha_bar) * noise_pred) / torch.sqrt(alpha)
            if step > 0:
                latents = latents + torch.sqrt(beta) * torch.randn_like(latents)
        return latents

