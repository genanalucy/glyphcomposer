#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.nn import functional as F

from glyphgen.config import load_yaml_config
from glyphgen.data.dataset import build_dataloader
from glyphgen.eval.probe import StructureProbe
from glyphgen.runtime import resolve_device
from glyphgen.structures import STRUCTURE_TYPES
from glyphgen.training.checkpoints import save_checkpoint
from glyphgen.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a structure classifier probe.")
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

    model = StructureProbe(num_classes=len(STRUCTURE_TYPES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 2e-4)))
    output_dir = Path(config.get("output_dir", "checkpoints/structure_probe"))
    best_val = 0.0

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        last_loss = 0.0
        for batch in train_loader:
            target = batch["structure_index"].to(device)
            glyphs = batch["target_image"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(glyphs)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        print(f"epoch={epoch} train_loss={last_loss:.4f}")

        if val_loader:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    logits = model(batch["target_image"].to(device))
                    prediction = logits.argmax(dim=1)
                    target = batch["structure_index"].to(device)
                    correct += int((prediction == target).sum().cpu())
                    total += int(target.numel())
            accuracy = correct / max(total, 1)
            print(f"epoch={epoch} val_acc={accuracy:.4f}")
            if accuracy >= best_val:
                best_val = accuracy
                save_checkpoint(
                    output_dir / "structure_probe_best.pt",
                    {
                        "model_type": "structure_probe",
                        "model_state": model.state_dict(),
                        "metadata": {"num_classes": len(STRUCTURE_TYPES), "config": config},
                    },
                )


if __name__ == "__main__":
    main()

