"""数据集路径解析：统一使用 ultralytics 的 datasets_dir 约定。

ultralytics 会把数据集 YAML 里的相对 ``path`` 解析到其配置的 ``datasets_dir``
（见 ``%APPDATA%/Ultralytics/settings.json``，本项目机器为 ``E:\\资料\\基座\\datasets``）
之下，而非项目根目录。因此本项目所有脚本与配置统一以 ``<datasets_dir>/ua_detrac``
作为数据集根目录，与 ``configs/ua_detrac.yaml`` 的 ``path: ua_detrac`` 保持一致。

ultralytics 未安装时回退到项目根下的 ./data（仅用于静态自检）。
"""

from __future__ import annotations

from pathlib import Path

DATASET_NAME = "ua_detrac"


def datasets_dir() -> Path:
    """返回 ultralytics 配置的 datasets 根目录；读取失败回退到 ./data。"""
    try:
        from ultralytics.utils import SETTINGS
        d = SETTINGS.get("datasets_dir")
        if d:
            return Path(d)
    except Exception:
        pass
    return Path("data")


def dataset_root(name: str = DATASET_NAME) -> Path:
    """返回数据集根目录：<datasets_dir>/<name>。"""
    return datasets_dir() / name
