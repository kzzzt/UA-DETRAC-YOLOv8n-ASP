"""SimAM 无参注意力模块（参数无关 3D 能量注意力）。

SimAM (A Simple, Parameter-Free Attention Module) 基于神经科学的
「空间抑制」现象，为每个空间位置计算能量函数并分配重要性权重，
不引入任何可学习参数，契合 YOLOv8n（nano）的轻量定位。

与本项目模型的接入方式：
    作为**独立层**接在每个检测头特征图（P2/P3/P4/P5）之后、Detect 之前，
    例如 YAML 中：

        - [15, 1, SimAM, []]   # 对第 15 层输出做注意力加权

    由于 SimAM 不改变通道数与分辨率，可直接通过 ultralytics 的通用模块
    解析路径构建（通道保持，c2=ch[f] 天然成立），无需特殊注册逻辑。

参考论文:
    Yang et al., "SimAM: A Simple, Parameter-Free Attention Module
    for Convolutional Neural Networks", ICML 2021.

实现说明（与官方 PyTorch 实现逐式对齐）:
    能量函数闭式解为
        e_t* = 4(σ^2 + λ) / ((t - μ)^2 + 2σ^2 + 2λ)
    注意力权重为 sigmoid(1/e_t*)。官方以等价形式
        d / (4(σ^2 + λ)) + 0.5,   d = (t - μ)^2
    实现；二者数学等价。方差 σ^2 采用分母 H*W-1（与官方一致）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimAM(nn.Module):
    """无参 3D 能量注意力模块。

    Args:
        e_lambda: 能量函数正则化系数 λ，默认 1e-4。
    """

    def __init__(self, e_lambda: float = 1e-4) -> None:
        super().__init__()
        self.e_lambda = e_lambda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入特征图，形状 (B, C, H, W)。

        Returns:
            注意力加权后的特征图，形状不变 (B, C, H, W)。
        """
        b, c, h, w = x.size()
        n = h * w - 1  # 除目标神经元外的神经元数量 M - 1

        mu = x.mean(dim=(2, 3), keepdim=True)
        d = (x - mu).pow(2)                                  # (t - μ)^2
        sigma_square = d.sum(dim=(2, 3), keepdim=True) / n   # 样本方差（分母 M-1）

        # 注意力权重 sigmoid(1/e*)，等价于官方 sigmoid(d / (4(σ^2+λ)) + 0.5)
        e_inv = d / (4.0 * (sigma_square + self.e_lambda)) + 0.5
        return x * self.sigmoid(e_inv)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lambda={self.e_lambda})"


class SimAM3D(nn.Module):
    """等价的教学对照实现（按 (B, C, H*W) 展开计算）。

    保留用于显式说明能量函数计算过程，与 :class:`SimAM` 语义一致。
    """

    def __init__(self, e_lambda: float = 1e-4) -> None:
        super().__init__()
        self.e_lambda = e_lambda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        n = h * w - 1
        x_flat = x.view(b, c, -1)
        mu = x_flat.mean(dim=-1, keepdim=True)
        d = (x_flat - mu).pow(2)
        sigma_square = d.sum(dim=-1, keepdim=True) / n
        e_inv = d / (4.0 * (sigma_square + self.e_lambda)) + 0.5
        return x * self.sigmoid(e_inv).view(b, c, h, w)


if __name__ == "__main__":
    module = SimAM()
    dummy = torch.randn(1, 64, 80, 80)
    out = module(dummy)
    assert out.shape == dummy.shape, f"shape mismatch: {out.shape}"
    print("SimAM forward OK, output shape:", tuple(out.shape))
