from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None

from glyphgen.models.components import (
    ConditionEncoder,
    ConvBlock,
    Downsample,
    Upsample,
    broadcast_condition,
)
from glyphgen.structures import STRUCTURE_TYPES


class ConditionalUNetBackbone(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 64) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.down1 = Downsample(base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.down2 = Downsample(base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.mid = ConvBlock(base_channels * 4, base_channels * 4)
        self.up2 = Upsample(base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.up1 = Upsample(base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 3, base_channels)
        self.out = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.layout_head = nn.Conv2d(base_channels, 2, kernel_size=1)

    def forward(self, x: "torch.Tensor") -> dict[str, "torch.Tensor"]:
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.down1(skip1))
        latent = self.mid(self.down2(skip2))
        x = self.up2(latent)
        x = self.dec2(torch.cat([x, skip2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, skip1], dim=1))
        return {
            "glyph": self.out(x).sigmoid(),
            "layout": self.layout_head(x),
        }


class GlyphBaselineModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        style_vocab_size: int,
        condition_embedding_dim: int = 128,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.condition_encoder = ConditionEncoder(
            vocab_size=vocab_size,
            style_vocab_size=style_vocab_size,
            structure_vocab_size=len(STRUCTURE_TYPES),
            embedding_dim=condition_embedding_dim,
        )
        self.backbone = ConditionalUNetBackbone(
            in_channels=2 + 2 + condition_embedding_dim,
            base_channels=base_channels,
        )

    def forward(
        self,
        *,
        component_images: "torch.Tensor",
        component_ids: "torch.Tensor",
        style_index: "torch.Tensor",
        structure_index: "torch.Tensor",
        layout_heatmap: "torch.Tensor",
    ) -> dict[str, "torch.Tensor"]:
        condition = self.condition_encoder(
            component_images=component_images,
            component_ids=component_ids,
            style_index=style_index,
            structure_index=structure_index,
        )
        condition_map = broadcast_condition(condition, layout_heatmap.shape[-2:])
        fused = torch.cat([component_images, layout_heatmap, condition_map], dim=1)
        return self.backbone(fused)

