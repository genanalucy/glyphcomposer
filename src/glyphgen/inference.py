from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from glyphgen.data.render import render_glyph_array
from glyphgen.models.baseline import GlyphBaselineModel
from glyphgen.models.diffusion import GlyphLatentDiffusion
from glyphgen.models.vae import GlyphVAE
from glyphgen.runtime import resolve_device
from glyphgen.structures import layout_heatmap, normalize_structure_label, structure_to_index
from glyphgen.training.checkpoints import load_checkpoint


def _tensor_from_image(path: str | Path, image_size: int) -> "torch.Tensor":
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required for inference.") from exc

    image = Image.open(path).convert("L").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _tensor_from_text(token: str, font_path: str | Path, image_size: int) -> "torch.Tensor":
    array = render_glyph_array(token, font_path, image_size)
    return torch.from_numpy(array).unsqueeze(0)


def prepare_condition_batch(
    *,
    component_a: str,
    component_b: str,
    structure: str,
    style_id: str,
    image_size: int,
    component_vocab: dict[str, int],
    style_vocab: dict[str, int],
    device: str,
    input_mode: str = "text_component",
    font_path: str | Path | None = None,
) -> dict[str, "torch.Tensor"]:
    if input_mode == "text_component":
        if font_path is None:
            raise ValueError("font_path is required for text_component inference.")
        component_a_image = _tensor_from_text(component_a, font_path, image_size)
        component_b_image = _tensor_from_text(component_b, font_path, image_size)
    else:
        component_a_image = _tensor_from_image(component_a, image_size)
        component_b_image = _tensor_from_image(component_b, image_size)

    structure_enum = normalize_structure_label(structure)
    component_ids = torch.tensor(
        [
            component_vocab.get(component_a, component_vocab.get("<unk>", 1)),
            component_vocab.get(component_b, component_vocab.get("<unk>", 1)),
        ],
        dtype=torch.long,
    ).unsqueeze(0)
    style_index = torch.tensor(
        [style_vocab.get(style_id, style_vocab.get("<unk>", 1))],
        dtype=torch.long,
    )
    structure_index = torch.tensor([structure_to_index(structure_enum)], dtype=torch.long)
    layout = torch.from_numpy(layout_heatmap(structure_enum, image_size)).unsqueeze(0)
    return {
        "component_images": torch.cat([component_a_image, component_b_image], dim=0).unsqueeze(0).to(device),
        "component_ids": component_ids.to(device),
        "style_index": style_index.to(device),
        "structure_index": structure_index.to(device),
        "layout_heatmap": layout.to(device),
    }


def load_model_bundle(
    *,
    checkpoint_path: str | Path,
    device: str = "auto",
    vae_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    device_name = resolve_device(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device_name)
    model_type = checkpoint["model_type"]
    metadata = checkpoint["metadata"]

    if model_type == "baseline":
        model = GlyphBaselineModel(
            vocab_size=metadata["vocab_size"],
            style_vocab_size=metadata["style_vocab_size"],
            condition_embedding_dim=metadata["condition_embedding_dim"],
            base_channels=metadata["base_channels"],
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(device_name).eval()
        return {
            "model_type": model_type,
            "model": model,
            "device": device_name,
            "image_size": metadata["image_size"],
            "component_vocab": metadata["component_vocab"],
            "style_vocab": metadata["style_vocab"],
        }

    if model_type == "diffusion":
        if vae_checkpoint_path is None:
            raise ValueError("vae_checkpoint_path is required for diffusion inference.")
        vae_checkpoint = load_checkpoint(vae_checkpoint_path, map_location=device_name)
        vae_meta = vae_checkpoint["metadata"]
        vae = GlyphVAE(
            latent_channels=vae_meta["latent_channels"],
            base_channels=vae_meta["base_channels"],
        )
        vae.load_state_dict(vae_checkpoint["model_state"])
        vae.to(device_name).eval()

        model = GlyphLatentDiffusion(
            vocab_size=metadata["vocab_size"],
            style_vocab_size=metadata["style_vocab_size"],
            latent_channels=metadata["latent_channels"],
            timesteps=metadata["timesteps"],
            condition_embedding_dim=metadata["condition_embedding_dim"],
            base_channels=metadata["base_channels"],
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(device_name).eval()
        return {
            "model_type": model_type,
            "model": model,
            "vae": vae,
            "device": device_name,
            "image_size": metadata["image_size"],
            "component_vocab": metadata["component_vocab"],
            "style_vocab": metadata["style_vocab"],
        }

    raise ValueError(f"Unsupported model_type: {model_type}")


def generate_from_bundle(bundle: dict[str, Any], batch: dict[str, "torch.Tensor"]) -> "torch.Tensor":
    with torch.no_grad():
        if bundle["model_type"] == "baseline":
            outputs = bundle["model"](**batch)
            return outputs["glyph"]

        if bundle["model_type"] == "diffusion":
            vae = bundle["vae"]
            model = bundle["model"]
            dummy = batch["component_images"].new_zeros(
                (batch["component_images"].shape[0], 1, bundle["image_size"], bundle["image_size"])
            )
            mu, _ = vae.encode(dummy)
            latents = model.sample(
                latent_shape=mu.shape,
                component_images=batch["component_images"],
                component_ids=batch["component_ids"],
                style_index=batch["style_index"],
                structure_index=batch["structure_index"],
                layout_heatmap=batch["layout_heatmap"],
            )
            return vae.decode(latents)

        raise ValueError(f"Unsupported model_type: {bundle['model_type']}")

