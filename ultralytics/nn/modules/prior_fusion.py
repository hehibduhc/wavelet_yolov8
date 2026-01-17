# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorStem(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, 3, stride=2, padding=1, bias=False),  # H/2
            nn.BatchNorm2d(out_ch // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch // 2, out_ch, 3, stride=2, padding=1, bias=False),  # H/4
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, priors: torch.Tensor) -> torch.Tensor:
        return self.net(priors)


class GatedAddFusion(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.gate = nn.Conv2d(ch * 2, ch, 1, bias=True)

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([x, p], dim=1)))
        return x + g * p


class PriorFusion(nn.Module):
    uses_priors = True

    def __init__(self, channels: int, prior_channels: int = 2, use_priors: bool = True):
        super().__init__()
        self.use_priors = use_priors
        if use_priors:
            self.stem = PriorStem(prior_channels, channels)
            self.fuse = GatedAddFusion(channels)

    def forward(self, x: torch.Tensor, priors: torch.Tensor | None = None) -> torch.Tensor:
        if not self.use_priors or priors is None:
            return x
        priors = priors.to(device=x.device, dtype=x.dtype)
        p = self.stem(priors)
        if p.shape[-2:] != x.shape[-2:]:
            p = F.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(x, p)
