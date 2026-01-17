# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .conv import Conv, autopad

try:  # pragma: no cover - optional dependency
    from torchvision.ops import deform_conv2d
except Exception as exc:  # pragma: no cover - optional dependency
    deform_conv2d = None
    _deform_conv2d_error = exc
else:  # pragma: no cover - optional dependency
    _deform_conv2d_error = None


def _to_ksize(k: int | tuple[int, int]) -> int:
    if isinstance(k, (tuple, list)):
        return int(k[0])
    return int(k)


class DSConv2d(nn.Module):
    """Dynamic Snake Convolution inspired 2D operator using offset-guided sampling."""

    def __init__(self, c1: int, c2: int, k: int | tuple[int, int] = 3, s: int = 1, p: int | None = None, g: int = 1):
        super().__init__()
        if deform_conv2d is None:
            raise ImportError(
                "torchvision.ops.deform_conv2d is required for DSConv2d but is not available."
            ) from _deform_conv2d_error
        k = _to_ksize(k)
        self.k = k
        self.stride = s
        self.padding = autopad(k, p)
        self.groups = g
        self.offset = nn.Conv2d(c1, 2 * k * k, kernel_size=3, stride=s, padding=1, bias=True)
        self.weight = nn.Parameter(torch.empty(c2, c1 // g, k, k))
        self.bias = nn.Parameter(torch.zeros(c2))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if deform_conv2d is None:
            raise ImportError(
                "torchvision.ops.deform_conv2d is required for DSConv2d but is not available."
            ) from _deform_conv2d_error
        offset = self.offset(x)
        return deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=1,
            groups=self.groups,
        )


class DSConvConv(nn.Module):
    """Drop-in Conv wrapper that switches between DSConv and standard Conv."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int | tuple[int, int] = 3,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        act: bool | nn.Module = True,
        use_dsconv: bool = True,
    ):
        super().__init__()
        self.use_dsconv = use_dsconv
        if not use_dsconv:
            self.conv = Conv(c1, c2, k, s, p, g, act)
            return
        if deform_conv2d is None:
            raise ImportError(
                "torchvision.ops.deform_conv2d is required for DSConvConv but is not available."
            ) from _deform_conv2d_error
        self.conv = DSConv2d(c1, c2, _to_ksize(k), s, p, g)
        self.bn = nn.BatchNorm2d(c2)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_dsconv:
            return self.conv(x)
        return self.act(self.bn(self.conv(x)))
