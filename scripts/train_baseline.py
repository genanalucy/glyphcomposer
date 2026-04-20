#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from glyphgen.config import load_yaml_config
from glyphgen.data.dataset import build_dataloader
from glyphgen.models.baseline import GlyphBaselineModel
from glyphgen.runtime import autocast_context, resolve_device
from glyphgen.training.checkpoints import save_checkpoint
from glyphgen.training.losses import (
    layout_supervision_loss,
    pyramid_perceptual_loss,
    reconstruction_loss,
    skeleton_consistency_loss,
)
from glyphgen.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline conditional U-Net.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val-manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))

    train_dataset, train_loader = build_dataloader(
        args.manifest,
        image_size=int(config["image_size"]),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
    )
    val_loader = None
    if args.val_manifest:
        _, val_loader = build_dataloader(
            args.val_manifest,
            image_size=int(config["image_size"]),
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config.get("num_workers", 0)),
            component_vocab=train_dataset.component_vocab,
            style_vocab=train_dataset.style_vocab,
        )

    model = GlyphBaselineModel(
        vocab_size=len(train_dataset.component_vocab.token_to_id),
        style_vocab_size=len(train_dataset.style_vocab.token_to_id),
        condition_embedding_dim=int(config.get("condition_embedding_dim", 128)),
        base_channels=int(config.get("base_channels", 64)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 2e-4)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))

    best_val = float("inf")
    output_dir = Path(config.get("output_dir", "checkpoints/baseline"))

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                outputs = model(
                    component_images=batch["component_images"],
                    component_ids=batch["component_ids"],
                    style_index=batch["style_index"],
                    structure_index=batch["structure_index"],
                    layout_heatmap=batch["layout_heatmap"],
                )
                loss = (
                    reconstruction_loss(outputs["glyph"], batch["target_image"]) * float(config.get("reconstruction_weight", 1.0))
                    + pyramid_perceptual_loss(outputs["glyph"], batch["target_image"]) * float(config.get("perceptual_weight", 0.1))
                    + skeleton_consistency_loss(outputs["glyph"], batch["target_image"]) * float(config.get("skeleton_weight", 0.2))
                    + layout_supervision_loss(outputs["layout"], batch["layout_heatmap"]) * float(config.get("layout_weight", 0.2))
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().cpu())

        mean_train = running / max(len(train_loader), 1)
        print(f"epoch={epoch} train_loss={mean_train:.4f}")

        mean_val = mean_train
        if val_loader:
            model.eval()
            total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
                    outputs = model(
                        component_images=batch["component_images"],
                        component_ids=batch["component_ids"],
                        style_index=batch["style_index"],
                        structure_index=batch["structure_index"],
                        layout_heatmap=batch["layout_heatmap"],
                    )
                    loss = reconstruction_loss(outputs["glyph"], batch["target_image"])
                    total += float(loss.detach().cpu())
            mean_val = total / max(len(val_loader), 1)
            print(f"epoch={epoch} val_recon={mean_val:.4f}")

        payload = {
            "model_type": "baseline",
            "model_state": model.state_dict(),
            "metadata": {
                "image_size": int(config["image_size"]),
                "base_channels": int(config.get("base_channels", 64)),
                "condition_embedding_dim": int(config.get("condition_embedding_dim", 128)),
                "vocab_size": len(train_dataset.component_vocab.token_to_id),
                "style_vocab_size": len(train_dataset.style_vocab.token_to_id),
                "component_vocab": train_dataset.component_vocab.token_to_id,
                "style_vocab": train_dataset.style_vocab.token_to_id,
                "config": config,
            },
        }
        save_checkpoint(output_dir / "baseline_last.pt", payload)
        if mean_val <= best_val:
            best_val = mean_val
            save_checkpoint(output_dir / "baseline_best.pt", payload)


if __name__ == "__main__":
    main()

