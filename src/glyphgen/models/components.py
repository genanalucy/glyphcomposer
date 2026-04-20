from __future__ import annotations

import math


try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError:  # pragma: no cover - exercised only when torch is missing
    torch = None
    nn = None
    F = None


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=max(1, min(8, out_channels)), num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, min(8, out_channels)), num_channels=out_channels),
            nn.SiLU(),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.block(x) + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.op(x)


class ComponentImageEncoder(nn.Module):
    def __init__(self, input_channels: int = 2, hidden_channels: int = 32, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(input_channels, hidden_channels),
            Downsample(hidden_channels),
            ConvBlock(hidden_channels, hidden_channels * 2),
            Downsample(hidden_channels * 2),
            ConvBlock(hidden_channels * 2, hidden_channels * 4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels * 4, out_dim),
            nn.SiLU(),
        )

    def forward(self, component_images: "torch.Tensor") -> "torch.Tensor":
        return self.net(component_images)


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        style_vocab_size: int,
        structure_vocab_size: int,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        half_dim = embedding_dim // 2
        self.component_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.style_embedding = nn.Embedding(style_vocab_size, half_dim)
        self.structure_embedding = nn.Embedding(structure_vocab_size, half_dim)
        self.image_encoder = ComponentImageEncoder(out_dim=embedding_dim)
        self.project = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(
        self,
        *,
        component_images: "torch.Tensor",
        component_ids: "torch.Tensor",
        style_index: "torch.Tensor",
        structure_index: "torch.Tensor",
    ) -> "torch.Tensor":
        text_embed = self.component_embedding(component_ids).flatten(start_dim=1)
        image_embed = self.image_encoder(component_images)
        style_embed = self.style_embedding(style_index)
        structure_embed = self.structure_embedding(structure_index)
        fused = torch.cat([text_embed, image_embed, style_embed, structure_embed], dim=1)
        return self.project(fused)


def broadcast_condition(condition: "torch.Tensor", spatial_size: tuple[int, int]) -> "torch.Tensor":
    return condition[..., None, None].expand(-1, -1, spatial_size[0], spatial_size[1])


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: "torch.Tensor") -> "torch.Tensor":
        half_dim = self.dim // 2
        frequency = math.log(10000) / max(half_dim - 1, 1)
        steps = torch.exp(torch.arange(half_dim, device=timesteps.device) * -frequency)
        angles = timesteps[:, None].float() * steps[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding
