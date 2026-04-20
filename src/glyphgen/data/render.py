from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from glyphgen.structures import Box


@lru_cache(maxsize=256)
def _cached_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, font_size)


def _fit_font_size(text: str, font_path: str | Path, target_box: Box, padding_ratio: float) -> int:
    usable_width = max(8, int(target_box.width * (1.0 - padding_ratio)))
    usable_height = max(8, int(target_box.height * (1.0 - padding_ratio)))
    max_size = max(12, min(target_box.width, target_box.height) * 2)
    min_size = 8

    dummy_image = Image.new("L", (target_box.width, target_box.height), 0)
    drawer = ImageDraw.Draw(dummy_image)
    best = min_size
    for font_size in range(max_size, min_size - 1, -2):
        font = _cached_font(str(font_path), font_size)
        bbox = drawer.textbbox((0, 0), text, font=font, anchor="lt")
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= usable_width and height <= usable_height:
            best = font_size
            break
    return best


def render_character(
    text: str,
    font_path: str | Path,
    image_size: int,
    *,
    box: Box | None = None,
    padding_ratio: float = 0.14,
    foreground: int = 255,
    background: int = 0,
) -> Image.Image:
    canvas = Image.new("L", (image_size, image_size), color=background)
    if box is None:
        box = Box(0, 0, image_size, image_size)
    font_size = _fit_font_size(text, font_path, box, padding_ratio)
    font = _cached_font(str(font_path), font_size)
    drawer = ImageDraw.Draw(canvas)
    center_x = (box.x0 + box.x1) / 2.0
    center_y = (box.y0 + box.y1) / 2.0
    drawer.text((center_x, center_y), text, font=font, fill=foreground, anchor="mm")
    return canvas


def render_glyph_array(
    text: str,
    font_path: str | Path,
    image_size: int,
    *,
    box: Box | None = None,
    padding_ratio: float = 0.14,
) -> "numpy.ndarray":
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required for glyph rendering.") from exc

    image = render_character(
        text,
        font_path,
        image_size,
        box=box,
        padding_ratio=padding_ratio,
    )
    return np.asarray(image, dtype=np.float32) / 255.0


def save_grayscale_image(path: str | Path, array: "numpy.ndarray") -> Path:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required for image saving.") from exc

    clipped = np.clip(array * 255.0, 0.0, 255.0).astype("uint8")
    image = Image.fromarray(clipped, mode="L")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path

