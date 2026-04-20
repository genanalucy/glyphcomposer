from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class StructureType(str, Enum):
    LEFT_RIGHT = "left_right"
    TOP_BOTTOM = "top_bottom"
    FULL_SURROUND = "full_surround"
    SURROUND_FROM_ABOVE = "surround_from_above"
    SURROUND_FROM_BELOW = "surround_from_below"
    SURROUND_FROM_LEFT = "surround_from_left"
    SURROUND_FROM_UPPER_LEFT = "surround_from_upper_left"
    SURROUND_FROM_UPPER_RIGHT = "surround_from_upper_right"
    SURROUND_FROM_LOWER_LEFT = "surround_from_lower_left"


class InputMode(str, Enum):
    TEXT_COMPONENT = "text_component"
    IMAGE_COMPONENT = "image_component"


@dataclass(slots=True)
class GlyphCondition:
    component_inputs: tuple[str, str]
    structure: StructureType
    style_id: str = "print"
    input_mode: InputMode = InputMode.TEXT_COMPONENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_inputs": list(self.component_inputs),
            "structure": self.structure.value,
            "style_id": self.style_id,
            "input_mode": self.input_mode.value,
        }


@dataclass(slots=True)
class GlyphSample:
    target_char: str
    font_id: str
    glyph_image: str
    structure: StructureType
    component_a: str
    component_b: str
    component_a_image: str
    component_b_image: str
    style_id: str = "print"
    split: str = "train_random"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def glyph_path(self) -> Path:
        return Path(self.glyph_image)

    def to_condition(self, input_mode: InputMode = InputMode.TEXT_COMPONENT) -> GlyphCondition:
        if input_mode is InputMode.TEXT_COMPONENT:
            component_inputs = (self.component_a, self.component_b)
        else:
            component_inputs = (self.component_a_image, self.component_b_image)
        return GlyphCondition(
            component_inputs=component_inputs,
            structure=self.structure,
            style_id=self.style_id,
            input_mode=input_mode,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "target_char": self.target_char,
            "font_id": self.font_id,
            "glyph_image": self.glyph_image,
            "structure": self.structure.value,
            "component_a": self.component_a,
            "component_b": self.component_b,
            "component_a_image": self.component_a_image,
            "component_b_image": self.component_b_image,
            "style_id": self.style_id,
            "split": self.split,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GlyphSample":
        return cls(
            target_char=payload["target_char"],
            font_id=payload["font_id"],
            glyph_image=payload["glyph_image"],
            structure=StructureType(payload["structure"]),
            component_a=payload["component_a"],
            component_b=payload["component_b"],
            component_a_image=payload["component_a_image"],
            component_b_image=payload["component_b_image"],
            style_id=payload.get("style_id", "print"),
            split=payload.get("split", "train_random"),
            metadata=payload.get("metadata", {}),
        )

