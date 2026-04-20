from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from glyphgen.data.manifests import read_manifest
from glyphgen.structures import layout_heatmap, structure_to_index
from glyphgen.types import GlyphSample


@dataclass(slots=True)
class Vocabulary:
    token_to_id: dict[str, int]

    @property
    def id_to_token(self) -> dict[int, str]:
        return {idx: token for token, idx in self.token_to_id.items()}

    def encode(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id["<unk>"])

    @classmethod
    def build(cls, tokens: list[str]) -> "Vocabulary":
        unique = ["<pad>", "<unk>"] + sorted(set(tokens))
        return cls(token_to_id={token: idx for idx, token in enumerate(unique)})


def _load_grayscale_tensor(path: str | Path, image_size: int) -> "torch.Tensor":
    try:
        import numpy as np
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch and numpy are required to load datasets.") from exc

    image = Image.open(path).convert("L").resize((image_size, image_size))
    data = torch.from_numpy(np.asarray(image)).float() / 255.0
    return data.unsqueeze(0)


class GlyphTensorDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int,
        *,
        component_vocab: Vocabulary | None = None,
        style_vocab: Vocabulary | None = None,
    ) -> None:
        try:
            import torch.utils.data  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError("torch is required to use GlyphTensorDataset.") from exc

        self.samples = read_manifest(manifest_path)
        self.image_size = image_size
        component_tokens = [sample.component_a for sample in self.samples] + [
            sample.component_b for sample in self.samples
        ]
        style_tokens = [sample.style_id for sample in self.samples]
        self.component_vocab = component_vocab or Vocabulary.build(component_tokens)
        self.style_vocab = style_vocab or Vocabulary.build(style_tokens)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, "torch.Tensor | str"]:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("torch is required to use GlyphTensorDataset.") from exc

        sample = self.samples[index]
        target_image = _load_grayscale_tensor(sample.glyph_image, self.image_size)
        component_a = _load_grayscale_tensor(sample.component_a_image, self.image_size)
        component_b = _load_grayscale_tensor(sample.component_b_image, self.image_size)
        layout = torch.from_numpy(layout_heatmap(sample.structure, self.image_size))

        return {
            "target_image": target_image,
            "component_images": torch.cat([component_a, component_b], dim=0),
            "component_ids": torch.tensor(
                [
                    self.component_vocab.encode(sample.component_a),
                    self.component_vocab.encode(sample.component_b),
                ],
                dtype=torch.long,
            ),
            "style_index": torch.tensor(self.style_vocab.encode(sample.style_id), dtype=torch.long),
            "structure_index": torch.tensor(structure_to_index(sample.structure), dtype=torch.long),
            "layout_heatmap": layout,
            "target_char": sample.target_char,
            "font_id": sample.font_id,
        }


def build_dataloader(
    manifest_path: str | Path,
    *,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    component_vocab: Vocabulary | None = None,
    style_vocab: Vocabulary | None = None,
) -> tuple[GlyphTensorDataset, "torch.utils.data.DataLoader"]:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required to build dataloaders.") from exc

    dataset = GlyphTensorDataset(
        manifest_path,
        image_size=image_size,
        component_vocab=component_vocab,
        style_vocab=style_vocab,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def read_samples(manifest_path: str | Path) -> list[GlyphSample]:
    return read_manifest(manifest_path)

