from __future__ import annotations

try:
    import torch
    from torch.nn import functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    F = None


def reconstruction_loss(prediction: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    return F.l1_loss(prediction, target)


def pyramid_perceptual_loss(prediction: "torch.Tensor", target: "torch.Tensor", levels: int = 3) -> "torch.Tensor":
    total = prediction.new_tensor(0.0)
    current_prediction = prediction
    current_target = target
    used_levels = 0
    for _ in range(levels):
        total = total + F.l1_loss(current_prediction, current_target)
        used_levels += 1
        if min(current_prediction.shape[-2:]) <= 8:
            break
        current_prediction = F.avg_pool2d(current_prediction, kernel_size=2)
        current_target = F.avg_pool2d(current_target, kernel_size=2)
    return total / max(used_levels, 1)


def _soft_skeleton(x: "torch.Tensor", iterations: int = 8) -> "torch.Tensor":
    skeleton = torch.zeros_like(x)
    current = x
    for _ in range(iterations):
        min_pool = -F.max_pool2d(-current, kernel_size=3, stride=1, padding=1)
        contour = F.relu(current - min_pool)
        skeleton = torch.maximum(skeleton, contour)
        current = F.avg_pool2d(current, kernel_size=3, stride=1, padding=1)
    return skeleton


def skeleton_consistency_loss(prediction: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    pred_skeleton = _soft_skeleton(prediction)
    target_skeleton = _soft_skeleton(target)
    return F.l1_loss(pred_skeleton, target_skeleton)


def layout_supervision_loss(prediction: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    return F.binary_cross_entropy_with_logits(prediction, target)


def kl_divergence_loss(mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

