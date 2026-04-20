#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image

from glyphgen.data.manifests import write_manifest_bundle
from glyphgen.types import GlyphSample, StructureType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OOV image-conditioned manifests.")
    parser.add_argument("--csv", required=True, help="CSV following assets/oov_image_samples_template.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=128)
    return parser.parse_args()


def _standardize_image(src: str, dst: Path, image_size: int) -> str:
    image = Image.open(src).convert("L").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(dst)
    return str(dst)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    samples: list[GlyphSample] = []
    with Path(args.csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            target_id = row["target_id"].strip()
            style_id = row.get("style_id", "oov").strip() or "oov"
            font_id = row.get("font_id", "manual").strip() or "manual"
            glyph_image = _standardize_image(
                row["target_image"],
                output_dir / "oov_images" / f"{target_id}_target.png",
                args.image_size,
            )
            component_a_image = _standardize_image(
                row["component_a_image"],
                output_dir / "oov_components" / f"{target_id}_a.png",
                args.image_size,
            )
            component_b_image = _standardize_image(
                row["component_b_image"],
                output_dir / "oov_components" / f"{target_id}_b.png",
                args.image_size,
            )
            samples.append(
                GlyphSample(
                    target_char=target_id,
                    font_id=font_id,
                    glyph_image=glyph_image,
                    structure=StructureType(row["structure"].strip()),
                    component_a=row.get("component_a", "component_a").strip() or "component_a",
                    component_b=row.get("component_b", "component_b").strip() or "component_b",
                    component_a_image=component_a_image,
                    component_b_image=component_b_image,
                    style_id=style_id,
                    split="oov_extension",
                )
            )

    bundle = write_manifest_bundle(samples=[], output_dir=output_dir, seed=42, oov_samples=samples)
    print(f"OOV manifest written to {bundle.oov_extension}")


if __name__ == "__main__":
    main()

