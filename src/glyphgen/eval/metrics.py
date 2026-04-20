from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None
    F = None


def mean_absolute_error(prediction: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    return torch.mean(torch.abs(prediction - target))


def structural_similarity(prediction: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(prediction, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    sigma_x = F.avg_pool2d(prediction * prediction, kernel_size=3, stride=1, padding=1) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(target * target, kernel_size=3, stride=1, padding=1) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(prediction * target, kernel_size=3, stride=1, padding=1) - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    return torch.mean(numerator / denominator)


class OptionalLPIPS:
    def __init__(self) -> None:
        self.metric = None
        try:
            import lpips

            self.metric = lpips.LPIPS(net="alex")
        except Exception:
            self.metric = None

    def __call__(self, prediction: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor | None":
        if self.metric is None:
            return None
        pred = prediction.repeat(1, 3, 1, 1) * 2.0 - 1.0
        truth = target.repeat(1, 3, 1, 1) * 2.0 - 1.0
        return self.metric(pred, truth).mean()

