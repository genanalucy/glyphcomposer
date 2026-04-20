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
from glyphgen.models.vae import GlyphVAE
from glyphgen.runtime import autocast_context, resolve_device
from glyphgen.training.checkpoints import save_checkpoint
from glyphgen.training.losses import kl_divergence_loss, reconstruction_loss
from glyphgen.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a glyph VAE.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val-manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))

    _, train_loader = build_dataloader(
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
        )

    model = GlyphVAE(
        latent_channels=int(config.get("latent_channels", 4)),
        base_channels=int(config.get("base_channels", 64)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))
    output_dir = Path(config.get("output_dir", "checkpoints/vae"))
    best_val = float("inf")

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            target = batch["target_image"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                outputs = model(target)
                loss = (
                    reconstruction_loss(outputs["reconstruction"], target) * float(config.get("reconstruction_weight", 1.0))
                    + kl_divergence_loss(outputs["mu"], outputs["logvar"]) * float(config.get("kl_weight", 0.001))
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach().cpu())
        mean_train = total / max(len(train_loader), 1)
        print(f"epoch={epoch} train_loss={mean_train:.4f}")

        mean_val = mean_train
        if val_loader:
            model.eval()
            total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    target = batch["target_image"].to(device)
                    outputs = model(target)
                    total += float(reconstruction_loss(outputs["reconstruction"], target).detach().cpu())
            mean_val = total / max(len(val_loader), 1)
            print(f"epoch={epoch} val_recon={mean_val:.4f}")

        payload = {
            "model_type": "vae",
            "model_state": model.state_dict(),
            "metadata": {
                "image_size": int(config["image_size"]),
                "latent_channels": int(config.get("latent_channels", 4)),
                "base_channels": int(config.get("base_channels", 64)),
                "config": config,
            },
        }
        save_checkpoint(output_dir / "vae_last.pt", payload)
        if mean_val <= best_val:
            best_val = mean_val
            save_checkpoint(output_dir / "vae_best.pt", payload)


if __name__ == "__main__":
    main()

