#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image

from glyphgen.inference import generate_from_bundle, load_model_bundle, prepare_condition_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a glyph image from components and structure.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae-checkpoint")
    parser.add_argument("--component-a", required=True)
    parser.add_argument("--component-b", required=True)
    parser.add_argument("--structure", required=True)
    parser.add_argument("--style-id", default="print")
    parser.add_argument("--input-mode", choices=["text_component", "image_component"], default="text_component")
    parser.add_argument("--font-path", default="/System/Library/Fonts/Hiragino Sans GB.ttc")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(
        checkpoint_path=args.checkpoint,
        device=args.device,
        vae_checkpoint_path=args.vae_checkpoint,
    )
    batch = prepare_condition_batch(
        component_a=args.component_a,
        component_b=args.component_b,
        structure=args.structure,
        style_id=args.style_id,
        image_size=bundle["image_size"],
        component_vocab=bundle["component_vocab"],
        style_vocab=bundle["style_vocab"],
        device=bundle["device"],
        input_mode=args.input_mode,
        font_path=args.font_path,
    )
    prediction = generate_from_bundle(bundle, batch)[0, 0].detach().cpu().numpy()
    image = Image.fromarray(np.clip(prediction * 255.0, 0, 255).astype("uint8"), mode="L")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved glyph to {output_path}")


if __name__ == "__main__":
    main()

