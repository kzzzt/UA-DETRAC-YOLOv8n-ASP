"""YOLOv8n-ASP 自定义模块集合与 Ultralytics 注册入口。

包含:
    - SimAM / SimAM3D: 无参注意力（独立层，YAML 中直接引用）
    - C2fSimAM: 融入 SimAM 的 C2f（参考/可选实现，默认配置不使用）
    - P2Fusion: P2 小目标检测层特征融合辅助组件
    - EIoULoss / eiou_loss / patch_ultralytics_eiou: EIOU 边界框回归损失

注册机制:
    Ultralytics 解析 YAML 时通过 ``parse_model`` 的 ``globals()[m]`` 查找模块名。
    本包导入时会调用 :func:`register_modules`，把「独立无参层」``SimAM``
    注入 ``ultralytics.nn.tasks`` 命名空间，使 YAML 可直接写:

        - [15, 1, SimAM, []]

    由于 SimAM 不改变通道数/分辨率，会落入 ``parse_model`` 的通用分支
    （c2 = ch[f] 天然成立），无需任何 hack。``C2fSimAM`` / ``P2Fusion`` 为
    独立辅助组件，不在默认 YAML 中使用，因此不注册。
"""

from __future__ import annotations

from .simam import SimAM, SimAM3D
from .c2f_simam import C2fSimAM
from .p2_head import P2Fusion, Conv, Upsample
from .eiou_loss import EIoULoss, eiou_loss, patch_ultralytics_eiou

__all__ = [
    "SimAM",
    "SimAM3D",
    "C2fSimAM",
    "P2Fusion",
    "Conv",
    "Upsample",
    "EIoULoss",
    "eiou_loss",
    "patch_ultralytics_eiou",
    "register_modules",
]


def register_modules() -> dict[str, type]:
    """将默认 YAML 需要的自定义模块注册到 Ultralytics 命名空间。

    Ultralytics 的 ``parse_model`` 使用 ``globals()[m]`` 查找模块名，
    因此把类写入 ``ultralytics.nn.tasks`` 模块的全局字典即可被 YAML 中的
    ``SimAM`` 引用。

    Returns:
        {模块名: 类} 映射，便于调用方打印注册结果。
    """
    registry = {
        "SimAM": SimAM,
    }

    try:
        import ultralytics.nn.tasks as tasks
    except ImportError:
        # ultralytics 未安装，不影响模型类作为独立模块使用
        return registry

    for name, cls in registry.items():
        setattr(tasks, name, cls)

    return registry


# 导入本包时自动完成注册（幂等，重复设置无副作用）
register_modules()
