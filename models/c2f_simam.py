"""融入 SimAM 的 C2f 模块（C2fSimAM，参考/可选实现）。

以 Ultralytics 原生 C2f 为基类，在其投影输出后接入无参 SimAM，
结构上与原生 C2f 完全兼容（cv1 / m / cv2 同名），理论上可直接用
``C2fSimAM`` 替换 YAML 中的 ``C2f`` 且预训练权重可迁移。

> 重要说明：本项目**默认配置不依赖本模块**。默认采用更简洁、与
> ``docs/实验设计方案.md`` 2.1 节一致的「独立 SimAM 层」接入方式
> （见 :mod:`models.simam`），直接写在 YAML 检测头之前，可被 ultralytics
> 通用解析路径直接构建。C2fSimAM 保留作为等价参考实现。

在 YAML 中使用 C2fSimAM 的注意事项：
    ultralytics >= 8.3 的 ``parse_model`` 使用 ``frozenset``（base_modules /
    repeat_modules）判定模块是否自动补齐 (c1, c2, n) 并做宽度缩放，而
    ``C2fSimAM`` 不在其中。如需在 YAML 中启用，需将 ``C2fSimAM`` 手动加入
    ``ultralytics/nn/tasks.py`` 的 ``base_modules`` 与 ``repeat_modules`` 两个
    集合（各一行）。因此默认配置选择独立 SimAM 层，规避该侵入式改动。

无 ultralytics 依赖时回退到内置最小实现，仅用于独立单元自检。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .simam import SimAM

try:
    from ultralytics.nn.modules.block import C2f as _UltraC2f
except Exception:  # ultralytics 未安装
    _UltraC2f = None


class _Bottleneck(nn.Module):
    """回退用标准瓶颈（含 shortcut），对齐 YOLOv8 结构。"""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 3, 1, 1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, 1, 1, groups=g, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.silu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.add:
            y = y + x
        return torch.nn.functional.silu(y)


class _FallbackC2f(nn.Module):
    """无 ultralytics 依赖时的最小 C2f 实现。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False,
                 g: int = 1, e: float = 0.5) -> None:
        super().__init__()
        from .p2_head import Conv

        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            _Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


_UltraC2f = _UltraC2f if _UltraC2f is not None else _FallbackC2f


class C2fSimAM(_UltraC2f):
    """融入 SimAM 无参注意力的 C2f 模块。

    在原生 C2f 输出（cv2 投影后）逐空间位置加权，抑制雨雾、夜间
    灯光等复杂背景干扰，不引入额外可学习参数。

    Args:
        c1: 输入通道数。
        c2: 输出通道数。
        n: 瓶颈重复次数（YAML 中的 depth）。
        shortcut: 是否使用残差连接。
        g: 分组卷积组数。
        e: 通道扩展系数（默认 0.5 对齐原生 C2f）。
        e_lambda: SimAM 能量函数正则化系数。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False,
                 g: int = 1, e: float = 0.5, e_lambda: float = 1e-4) -> None:
        super().__init__(c1, c2, n, shortcut, g, e)
        self.simam = SimAM(e_lambda)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.simam(super().forward(x))


if __name__ == "__main__":
    module = C2fSimAM(64, 64, n=2)
    dummy = torch.randn(1, 64, 80, 80)
    out = module(dummy)
    assert out.shape == dummy.shape, f"shape mismatch: {out.shape}"
    print("C2fSimAM forward OK, output shape:", tuple(out.shape))
