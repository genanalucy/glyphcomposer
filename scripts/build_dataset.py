#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glyphgen.data.decomposition import filter_two_component_records, load_decomposition_csv
from glyphgen.data.manifests import write_manifest_bundle
from glyphgen.data.render import render_glyph_array, save_grayscale_image
from glyphgen.types import GlyphSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a font-rendered structured glyph dataset.")
    parser.add_argument("--decomposition-csv", required=True)
    parser.add_argument("--font-path", action="append", required=True, help="Repeat for multiple fonts.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def render_records(
    *,
    records,
    font_path: str,
    output_dir: Path,
    image_size: int,
) -> list[GlyphSample]:
    font_id = Path(font_path).stem.replace(" ", "_")
    samples: list[GlyphSample] = []
    for record in records:
        glyph_path = output_dir / "images" / font_id / f"{record.target_char}_glyph.png"
        component_a_path = output_dir / "components" / font_id / f"{record.target_char}_a.png"
        component_b_path = output_dir / "components" / font_id / f"{record.target_char}_b.png"
        glyph_array = render_glyph_array(record.target_char, font_path, image_size)
        component_a_array = render_glyph_array(record.component_a, font_path, image_size)
        component_b_array = render_glyph_array(record.component_b, font_path, image_size)
        save_grayscale_image(glyph_path, glyph_array)
        save_grayscale_image(component_a_path, component_a_array)
        save_grayscale_image(component_b_path, component_b_array)
        samples.append(
            GlyphSample(
                target_char=record.target_char,
                font_id=font_id,
                glyph_image=str(glyph_path),
                structure=record.structure,
                component_a=record.component_a,
                component_b=record.component_b,
                component_a_image=str(component_a_path),
                component_b_image=str(component_b_path),
                style_id=font_id,
                split="main",
                metadata={"font_path": str(font_path)},
            )
        )
    return samples


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    base_records = filter_two_component_records(load_decomposition_csv(args.decomposition_csv))
    samples: list[GlyphSample] = []
    for font_path in args.font_path:
        samples.extend(
            render_records(
                records=base_records,
                font_path=font_path,
                output_dir=output_dir,
                image_size=args.image_size,
            )
        )

    paths = write_manifest_bundle(samples, output_dir, seed=args.seed)
    print("Dataset built successfully.")
    print(f"Main manifests written to: {paths.train_random.parent}")


if __name__ == "__main__":
    main()

