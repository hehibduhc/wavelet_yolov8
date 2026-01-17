# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, autopad


class DynamicSnakeConv(nn.Module):
    """Dynamic Snake Convolution from DSCNet (YaoleiQi/DSCNet)."""

    def __init__(self, inc: int, outc: int, kernel_size: int = 3, bias: bool = False):
        super().__init__()
        self.kernel_size = kernel_size

        # Offset prediction convolution: output 2 * kernel_size (x/y offsets).
        self.offset = nn.Conv2d(
            inc,
            2 * kernel_size,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )
        # Feature convolution: extract final features.
        self.conv_kernel = nn.Conv2d(
            inc,
            outc,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            bias=bias,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.offset.weight, 0.0)
        if self.offset.bias is not None:
            nn.init.constant_(self.offset.bias, 0.0)
        nn.init.kaiming_normal_(self.conv_kernel.weight, mode="fan_out", nonlinearity="relu")
        if self.conv_kernel.bias is not None:
            nn.init.constant_(self.conv_kernel.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset(x)
        dtype = offset.data.type()
        ks = self.kernel_size
        n, _, h, w = offset.size()

        # Core 1: generate coordinate grid.
        grid_y = torch.arange(-(ks - 1) // 2, (ks - 1) // 2 + 1).type(dtype)
        grid_y = grid_y.repeat(n, 1, h, w)
        grid_y = grid_y.reshape(n, 1, ks, h, w)

        # Core 2: snake-like iterative offsets.
        offset = torch.tanh(offset)
        offset = offset.reshape(n, 2, ks, h, w)
        offset_y = offset[:, 0, :, :, :]
        # offset_x is kept for parity with the official formulation, even if unused here.
        offset_x = offset[:, 1, :, :, :]

        y_center = (ks - 1) // 2
        y_offset_new = torch.zeros_like(offset_y)
        y_offset_new[:, y_center, :, :] = offset_y[:, y_center, :, :]
        for idx in range(1, y_center + 1):
            y_offset_new[:, y_center + idx, :, :] = (
                y_offset_new[:, y_center + idx - 1, :, :] + offset_y[:, y_center + idx, :, :]
            )
            y_offset_new[:, y_center - idx, :, :] = (
                y_offset_new[:, y_center - idx + 1, :, :] + offset_y[:, y_center - idx, :, :]
            )

        # Core 3: normalize coordinates and sample with grid_sample.
        grid_y = grid_y + y_offset_new.unsqueeze(1)
        grid_x = torch.zeros_like(grid_y)

        grid_y = grid_y.reshape(n, ks * h, w)
        grid_x = grid_x.reshape(n, ks * h, w)
        grid = torch.stack((grid_x, grid_y), dim=3)

        grid[:, :, :, 0] = grid[:, :, :, 0] / ((w - 1.0) / 2.0)
        grid[:, :, :, 1] = grid[:, :, :, 1] / ((h - 1.0) / 2.0)
        grid = grid - 1.0

        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        x = self.conv_kernel(x)

        return x


class DSConvConv(nn.Module):
    """Drop-in Conv wrapper to match YOLOv8 Conv signature with DSConv support."""

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
        self.stride = s
        k = k if isinstance(k, int) else int(k[0])

        if not use_dsconv:
            p = autopad(k, p)
            self.conv = Conv(c1, c2, k, s, p, g, act)
            return

        self.dsc = DynamicSnakeConv(c1, c2, kernel_size=k)
        self.bn = nn.BatchNorm2d(c2)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_dsconv:
            return self.conv(x)
        out = self.dsc(x)
        out = self.bn(out)
        out = self.act(out)
        return out if self.stride == 1 else F.max_pool2d(out, self.stride)
