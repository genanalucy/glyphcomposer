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
from glyphgen.models.diffusion import GlyphLatentDiffusion
from glyphgen.models.vae import GlyphVAE
from glyphgen.runtime import autocast_context, resolve_device
from glyphgen.training.checkpoints import load_checkpoint, save_checkpoint
from glyphgen.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent diffusion on VAE latents.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
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

    vae_checkpoint = load_checkpoint(args.vae_checkpoint, map_location=device)
    vae_meta = vae_checkpoint["metadata"]
    vae = GlyphVAE(
        latent_channels=vae_meta["latent_channels"],
        base_channels=vae_meta["base_channels"],
    ).to(device)
    vae.load_state_dict(vae_checkpoint["model_state"])
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad = False

    model = GlyphLatentDiffusion(
        vocab_size=len(train_dataset.component_vocab.token_to_id),
        style_vocab_size=len(train_dataset.style_vocab.token_to_id),
        latent_channels=int(config.get("latent_channels", 4)),
        timesteps=int(config.get("timesteps", 200)),
        condition_embedding_dim=int(config.get("condition_embedding_dim", 128)),
        base_channels=int(config.get("base_channels", 128)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))
    output_dir = Path(config.get("output_dir", "checkpoints/diffusion"))
    best_val = float("inf")

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                mu, _ = vae.encode(batch["target_image"])
            with autocast_context(device):
                loss, _ = model.training_loss(
                    latents=mu,
                    component_images=batch["component_images"],
                    component_ids=batch["component_ids"],
                    style_index=batch["style_index"],
                    structure_index=batch["structure_index"],
                    layout_heatmap=batch["layout_heatmap"],
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
                    batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
                    mu, _ = vae.encode(batch["target_image"])
                    loss, _ = model.training_loss(
                        latents=mu,
                        component_images=batch["component_images"],
                        component_ids=batch["component_ids"],
                        style_index=batch["style_index"],
                        structure_index=batch["structure_index"],
                        layout_heatmap=batch["layout_heatmap"],
                    )
                    total += float(loss.detach().cpu())
            mean_val = total / max(len(val_loader), 1)
            print(f"epoch={epoch} val_loss={mean_val:.4f}")

        payload = {
            "model_type": "diffusion",
            "model_state": model.state_dict(),
            "metadata": {
                "image_size": int(config["image_size"]),
                "latent_channels": int(config.get("latent_channels", 4)),
                "timesteps": int(config.get("timesteps", 200)),
                "condition_embedding_dim": int(config.get("condition_embedding_dim", 128)),
                "base_channels": int(config.get("base_channels", 128)),
                "vocab_size": len(train_dataset.component_vocab.token_to_id),
                "style_vocab_size": len(train_dataset.style_vocab.token_to_id),
                "component_vocab": train_dataset.component_vocab.token_to_id,
                "style_vocab": train_dataset.style_vocab.token_to_id,
                "config": config,
            },
        }
        save_checkpoint(output_dir / "diffusion_last.pt", payload)
        if mean_val <= best_val:
            best_val = mean_val
            save_checkpoint(output_dir / "diffusion_best.pt", payload)


if __name__ == "__main__":
    main()

