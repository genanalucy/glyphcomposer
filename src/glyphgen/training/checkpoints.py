from __future__ import annotations

from pathlib import Path


def save_checkpoint(path: str | Path, payload: dict) -> Path:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required to save checkpoints.") from exc

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(path: str | Path, map_location: str | None = None) -> dict:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required to load checkpoints.") from exc

    return torch.load(Path(path), map_location=map_location or "cpu")

