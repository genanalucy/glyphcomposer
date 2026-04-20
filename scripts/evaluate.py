#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from glyphgen.data.dataset import Vocabulary, build_dataloader
from glyphgen.eval.metrics import OptionalLPIPS, mean_absolute_error, structural_similarity
from glyphgen.eval.probe import StructureProbe
from glyphgen.inference import generate_from_bundle, load_model_bundle
from glyphgen.runtime import resolve_device
from glyphgen.training.checkpoints import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline or diffusion glyph generation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vae-checkpoint")
    parser.add_argument("--probe-checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", default="outputs/eval_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(
        checkpoint_path=args.checkpoint,
        device=args.device,
        vae_checkpoint_path=args.vae_checkpoint,
    )
    device = resolve_device(args.device)
    _, loader = build_dataloader(
        args.manifest,
        image_size=bundle["image_size"],
        batch_size=4,
        shuffle=False,
        num_workers=0,
        component_vocab=Vocabulary(bundle["component_vocab"]),
        style_vocab=Vocabulary(bundle["style_vocab"]),
    )

    probe = None
    if args.probe_checkpoint:
        checkpoint = load_checkpoint(args.probe_checkpoint, map_location=device)
        probe = StructureProbe(num_classes=checkpoint["metadata"]["num_classes"]).to(device)
        probe.load_state_dict(checkpoint["model_state"])
        probe.eval()

    lpips_metric = OptionalLPIPS()
    total_mae = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    lpips_count = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            target = batch["target_image"].to(device)
            model_batch = {
                "component_images": batch["component_images"].to(device),
                "component_ids": batch["component_ids"].to(device),
                "style_index": batch["style_index"].to(device),
                "structure_index": batch["structure_index"].to(device),
                "layout_heatmap": batch["layout_heatmap"].to(device),
            }
            prediction = generate_from_bundle(bundle, model_batch)
            total_mae += float(mean_absolute_error(prediction, target).cpu())
            total_ssim += float(structural_similarity(prediction, target).cpu())
            lpips_value = lpips_metric(prediction.cpu(), target.cpu())
            if lpips_value is not None:
                total_lpips += float(lpips_value.cpu())
                lpips_count += 1
            if probe is not None:
                logits = probe(prediction)
                correct += int((logits.argmax(dim=1) == batch["structure_index"].to(device)).sum().cpu())
            total += int(target.shape[0])

    metrics = {
        "mae": total_mae / max(total, 1),
        "ssim": total_ssim / max(total, 1),
    }
    if lpips_count:
        metrics["lpips"] = total_lpips / lpips_count
    if probe is not None:
        metrics["structure_accuracy"] = correct / max(total, 1)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
