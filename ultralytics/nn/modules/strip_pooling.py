# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F


class StripPooling(nn.Module):
    """
    Strip Pooling core: aggregate long-range context by pooling along one spatial dimension.
    """

    def __init__(self, channels: int, reduction: int = 4, norm=nn.BatchNorm2d):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            norm(mid),
            nn.ReLU(inplace=True),
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=(3, 1), padding=(1, 0), bias=False),
            norm(mid),
            nn.ReLU(inplace=True),
        )
        self.conv_w = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=(1, 3), padding=(0, 1), bias=False),
            norm(mid),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(mid, channels, kernel_size=1, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.conv1(x)

        # pool along width -> (B, mid, H, 1)
        y_h = F.adaptive_avg_pool2d(y, (h, 1))
        y_h = self.conv_h(y_h)
        y_h = F.interpolate(y_h, size=(h, w), mode="bilinear", align_corners=False)

        # pool along height -> (B, mid, 1, W)
        y_w = F.adaptive_avg_pool2d(y, (1, w))
        y_w = self.conv_w(y_w)
        y_w = F.interpolate(y_w, size=(h, w), mode="bilinear", align_corners=False)

        out = y_h + y_w
        out = self.conv2(out)
        return out


class StripPoolingAttn(nn.Module):
    """
    Residual attention wrapper: x * sigmoid(StripPooling(x)) + x
    """

    def __init__(self, channels: int, enabled: bool = True, reduction: int = 4, norm=nn.BatchNorm2d):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.sp = StripPooling(channels, reduction=reduction, norm=norm)
            self.act = nn.Sigmoid()

    def forward(self, x):
        if not self.enabled:
            return x
        a = self.act(self.sp(x))
        return x * a + x
