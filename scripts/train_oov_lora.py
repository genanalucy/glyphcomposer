#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from glyphgen.config import load_yaml_config
from glyphgen.data.dataset import Vocabulary, build_dataloader
from glyphgen.models.diffusion import GlyphLatentDiffusion
from glyphgen.models.lora import inject_lora, mark_only_lora_trainable
from glyphgen.models.vae import GlyphVAE
from glyphgen.runtime import autocast_context, resolve_device
from glyphgen.training.checkpoints import load_checkpoint, save_checkpoint
from glyphgen.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA adaptation for OOV glyph generation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--diffusion-checkpoint", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))

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

    diffusion_checkpoint = load_checkpoint(args.diffusion_checkpoint, map_location=device)
    metadata = diffusion_checkpoint["metadata"]
    model = GlyphLatentDiffusion(
        vocab_size=metadata["vocab_size"],
        style_vocab_size=metadata["style_vocab_size"],
        latent_channels=metadata["latent_channels"],
        timesteps=metadata["timesteps"],
        condition_embedding_dim=metadata["condition_embedding_dim"],
        base_channels=metadata["base_channels"],
    ).to(device)
    model.load_state_dict(diffusion_checkpoint["model_state"])
    train_dataset, train_loader = build_dataloader(
        args.manifest,
        image_size=int(config["image_size"]),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        component_vocab=Vocabulary(metadata["component_vocab"]),
        style_vocab=Vocabulary(metadata["style_vocab"]),
    )
    inject_lora(
        model,
        rank=int(config.get("lora_rank", 8)),
        alpha=int(config.get("lora_alpha", 16)),
        dropout=float(config.get("lora_dropout", 0.05)),
    )
    mark_only_lora_trainable(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config.get("learning_rate", 5e-5)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))
    output_dir = Path(config.get("output_dir", "checkpoints/oov_lora"))

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
        print(f"epoch={epoch} train_loss={total / max(len(train_loader), 1):.4f}")

        save_checkpoint(
            output_dir / "oov_lora_last.pt",
            {
                "model_type": "diffusion",
                "model_state": model.state_dict(),
                "metadata": metadata | {"lora_config": config},
            },
        )


if __name__ == "__main__":
    main()
