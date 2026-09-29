"""P2 小目标检测层与特征融合辅助模块。

YOLOv8 默认检测头为 P3/P4/P5（下采样 8/16/32 倍）。为提升远处
小尺寸车辆检测能力，本模块从骨干更浅层（P2，下采样 4 倍，
即 160×160 @ 640 输入）引出高分辨率特征并接入检测头。

该模块以组件形式提供，便于在自定义模型 YAML 中组合使用。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Conv(nn.Module):
    """标准卷积块：Conv2d -> BatchNorm2d -> SiLU。"""

    def __init__(self, in_channels: int, out_channels: int, k: int = 1,
                 s: int = 1, p: int | None = None) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Upsample(nn.Module):
    """最近邻上采样，用于 P3 特征向 P2 尺度对齐。"""

    def __init__(self, scale_factor: int = 2) -> None:
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode="nearest"
        )


class P2Fusion(nn.Module):
    """构建 P2 输出特征：浅层 P2 主干特征 + P3 上采样特征。

    典型流程：
        p2 = backbone 输出的 4 倍下采样特征 (C2)
        p3 = neck 输出的 8 倍下采样特征 (P3)
        p3_up = Upsample(p3)            # 至 4 倍下采样尺度
        p2_out = Concat + Conv(p2, p3_up)

    Args:
        c2_channels: 浅层 P2 特征通道数。
        p3_channels: 上层 P3 特征通道数。
        out_channels: 融合后输出通道数。
    """

    def __init__(self, c2_channels: int, p3_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = Upsample(scale_factor=2)
        self.conv = Conv(c2_channels + p3_channels, out_channels, 1, 1)

    def forward(self, c2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        p3_up = self.upsample(p3)
        out = torch.cat([c2, p3_up], dim=1)
        return self.conv(out)


if __name__ == "__main__":
    fusion = P2Fusion(c2_channels=128, p3_channels=256, out_channels=128)
    c2 = torch.randn(1, 128, 160, 160)
    p3 = torch.randn(1, 256, 80, 80)
    out = fusion(c2, p3)
    assert out.shape == (1, 128, 160, 160), f"shape mismatch: {out.shape}"
    print("P2Fusion forward OK, output shape:", tuple(out.shape))