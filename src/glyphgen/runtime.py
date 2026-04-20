from __future__ import annotations

from contextlib import nullcontext


def resolve_device(device: str = "auto") -> str:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required to resolve devices.") from exc

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def autocast_context(device: str):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for autocast.") from exc

    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

