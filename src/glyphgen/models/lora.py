from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int = 8, alpha: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.down = nn.Conv2d(base.in_channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.base(x) + self.up(self.dropout(self.down(x))) * self.scale


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.base(x) + self.up(self.dropout(self.down(x))) * self.scale


def inject_lora(module: nn.Module, *, rank: int = 8, alpha: int = 16, dropout: float = 0.0) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and child.kernel_size == (1, 1):
            setattr(module, name, LoRAConv2d(child, rank=rank, alpha=alpha, dropout=dropout))
        elif isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
        else:
            inject_lora(child, rank=rank, alpha=alpha, dropout=dropout)
    return module


def mark_only_lora_trainable(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False
    for child in module.modules():
        if isinstance(child, (LoRAConv2d, LoRALinear)):
            for parameter in child.parameters():
                parameter.requires_grad = True

