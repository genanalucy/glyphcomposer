from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .types import StructureType


@dataclass(frozen=True, slots=True)
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


STRUCTURE_TYPES: tuple[StructureType, ...] = tuple(StructureType)


def normalize_structure_label(raw: str) -> StructureType:
    return StructureType(raw.strip().lower())


def list_structure_labels() -> list[str]:
    return [item.value for item in STRUCTURE_TYPES]


def component_boxes(structure: StructureType, image_size: int, margin_ratio: float = 0.08) -> tuple[Box, Box]:
    margin = max(2, int(image_size * margin_ratio))
    inner_left = margin
    inner_top = margin
    inner_right = image_size - margin
    inner_bottom = image_size - margin
    mid_x = image_size // 2
    mid_y = image_size // 2

    if structure is StructureType.LEFT_RIGHT:
        return (
            Box(inner_left, inner_top, mid_x, inner_bottom),
            Box(mid_x, inner_top, inner_right, inner_bottom),
        )
    if structure is StructureType.TOP_BOTTOM:
        return (
            Box(inner_left, inner_top, inner_right, mid_y),
            Box(inner_left, mid_y, inner_right, inner_bottom),
        )
    if structure is StructureType.FULL_SURROUND:
        return (
            Box(inner_left, inner_top, inner_right, inner_bottom),
            Box(margin * 2, margin * 2, image_size - margin * 2, image_size - margin * 2),
        )
    if structure is StructureType.SURROUND_FROM_ABOVE:
        return (
            Box(inner_left, inner_top, inner_right, image_size - margin),
            Box(margin * 2, image_size // 3, image_size - margin * 2, image_size - margin * 2),
        )
    if structure is StructureType.SURROUND_FROM_BELOW:
        return (
            Box(inner_left, margin, inner_right, inner_bottom),
            Box(margin * 2, margin * 2, image_size - margin * 2, image_size * 2 // 3),
        )
    if structure is StructureType.SURROUND_FROM_LEFT:
        return (
            Box(inner_left, inner_top, inner_right, inner_bottom),
            Box(image_size // 3, margin * 2, image_size - margin * 2, image_size - margin * 2),
        )
    if structure is StructureType.SURROUND_FROM_UPPER_LEFT:
        return (
            Box(inner_left, inner_top, inner_right, inner_bottom),
            Box(image_size // 3, image_size // 3, image_size - margin * 2, image_size - margin * 2),
        )
    if structure is StructureType.SURROUND_FROM_UPPER_RIGHT:
        return (
            Box(inner_left, inner_top, inner_right, inner_bottom),
            Box(margin * 2, image_size // 3, image_size * 2 // 3, image_size - margin * 2),
        )
    if structure is StructureType.SURROUND_FROM_LOWER_LEFT:
        return (
            Box(inner_left, inner_top, inner_right, inner_bottom),
            Box(image_size // 3, margin * 2, image_size - margin * 2, image_size * 2 // 3),
        )
    raise ValueError(f"Unsupported structure: {structure}")


def layout_heatmap(structure: StructureType, image_size: int) -> "numpy.ndarray":
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required for layout heatmap generation.") from exc

    heatmap = np.zeros((2, image_size, image_size), dtype=np.float32)
    for idx, box in enumerate(component_boxes(structure, image_size)):
        heatmap[idx, box.y0:box.y1, box.x0:box.x1] = 1.0
    return heatmap


def structure_to_index(structure: StructureType) -> int:
    return STRUCTURE_TYPES.index(structure)


def index_to_structure(index: int) -> StructureType:
    return STRUCTURE_TYPES[index]


def structure_names(values: Iterable[StructureType]) -> list[str]:
    return [item.value for item in values]

