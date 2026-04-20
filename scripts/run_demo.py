#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
import numpy as np

from glyphgen.inference import generate_from_bundle, load_model_bundle, prepare_condition_batch
from glyphgen.structures import list_structure_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the GlyphGen Gradio demo.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae-checkpoint")
    parser.add_argument("--font-path", default="/System/Library/Fonts/Hiragino Sans GB.ttc")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(
        checkpoint_path=args.checkpoint,
        device=args.device,
        vae_checkpoint_path=args.vae_checkpoint,
    )

    def infer(component_a: str, component_b: str, structure: str, style_id: str):
        batch = prepare_condition_batch(
            component_a=component_a,
            component_b=component_b,
            structure=structure,
            style_id=style_id or "print",
            image_size=bundle["image_size"],
            component_vocab=bundle["component_vocab"],
            style_vocab=bundle["style_vocab"],
            device=bundle["device"],
            input_mode="text_component",
            font_path=args.font_path,
        )
        prediction = generate_from_bundle(bundle, batch)[0, 0].detach().cpu().numpy()
        return np.clip(prediction, 0.0, 1.0)

    demo = gr.Interface(
        fn=infer,
        inputs=[
            gr.Textbox(label="Component A", value="犭"),
            gr.Textbox(label="Component B", value="句"),
            gr.Dropdown(choices=list_structure_labels(), label="Structure", value="left_right"),
            gr.Textbox(label="Style ID", value="print"),
        ],
        outputs=gr.Image(label="Generated Glyph", type="numpy", image_mode="L"),
        title="GlyphGen Demo",
        description="Generate glyph images from component pairs and layout structures.",
    )
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()

