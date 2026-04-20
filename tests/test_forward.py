import pytest

torch = pytest.importorskip("torch")

from glyphgen.models.baseline import GlyphBaselineModel
from glyphgen.models.vae import GlyphVAE


def test_baseline_forward_shape() -> None:
    model = GlyphBaselineModel(
        vocab_size=8,
        style_vocab_size=4,
        condition_embedding_dim=32,
        base_channels=16,
    )
    outputs = model(
        component_images=torch.randn(2, 2, 64, 64),
        component_ids=torch.tensor([[2, 3], [4, 5]], dtype=torch.long),
        style_index=torch.tensor([1, 2], dtype=torch.long),
        structure_index=torch.tensor([0, 1], dtype=torch.long),
        layout_heatmap=torch.randn(2, 2, 64, 64),
    )
    assert outputs["glyph"].shape == (2, 1, 64, 64)
    assert outputs["layout"].shape == (2, 2, 64, 64)


def test_vae_forward_shape() -> None:
    model = GlyphVAE(latent_channels=4, base_channels=16)
    inputs = torch.randn(2, 1, 64, 64)
    outputs = model(inputs)
    assert outputs["reconstruction"].shape == inputs.shape
