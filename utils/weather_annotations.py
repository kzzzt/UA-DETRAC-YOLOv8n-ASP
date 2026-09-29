"""天气/光照标签映射与场景划分工具。

UA-DETRAC 官方未对每一帧提供统一天气标签，但序列（sequence）层面
存在可辨识的场景特征。本模块提供:

1. 场景类型常量与中文/英文映射。
2. 基于序列 ID 与官方附带的场景信息（若存在）的归类函数。
3. 占位/兜底规则：当无外部标注时，按序列 ID 均分到四类，
   保证脚本可运行，用户可替换为真实天气标注。

场景类型:
    sunny  - 晴天
    cloudy - 阴天
    night  - 夜间
    rainy  - 雨天
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

# 场景常量
SCENES: tuple[str, ...] = ("sunny", "cloudy", "night", "rainy")

SCENE_CN: dict[str, str] = {
    "sunny": "晴天",
    "cloudy": "阴天",
    "night": "夜间",
    "rainy": "雨天",
}

SCENE_EN: dict[str, str] = {v: k for k, v in SCENE_CN.items()}

# 别名归一化：启发式或外部标注可能产出 day/rain/sun 等非规范写法
SCENE_ALIASES: dict[str, str] = {
    "day": "sunny",
    "sun": "sunny",
    "clear": "sunny",
    "sunny": "sunny",
    "rain": "rainy",
    "rainy": "rainy",
    "cloudy": "cloudy",
    "overcast": "cloudy",
    "night": "night",
    "dark": "night",
}

# 默认兜底规则：若无可用的真实天气标注，按序列 ID 取模划分。
# 此为占位逻辑，论文实验必须替换为官方附件或人工核验的天气标签。
FALLBACK_MOD: int = 4


def normalize_scene(name: str) -> str:
    """将中英文场景名归一化为标准 key（sunny/cloudy/night/rainy）。

    支持英文标准名、常见别名（day/rain/sun/overcast/dark 等）与中文名。
    无法识别时返回原输入的小写形式（由调用方决定是否丢弃）。

    Args:
        name: 场景名（如 "晴天"、"rainy"、"day"、"night"）。

    Returns:
        归一化后的英文 key。
    """
    if not name:
        return ""
    stripped = name.strip()
    key = stripped.lower()
    if key in SCENE_ALIASES:
        return SCENE_ALIASES[key]
    if stripped in SCENE_CN.values():  # 中文名：晴天/阴天/夜间/雨天
        return SCENE_EN[stripped]
    return key


def load_scene_map(index_file: str | os.PathLike | None) -> dict[str, str]:
    """从外部文件加载「序列/图像 ID → 场景」映射。

    文件格式（每行一条）：
        <image_or_sequence_id><sep>sunny|cloudy|night|rainy
    其中 <sep> 为空白字符（空格、Tab 或逗号）。

    Args:
        index_file: 场景标注文件路径；若为 None 则返回空字典。

    Returns:
        映射字典 {id: scene}。
    """
    if index_file is None:
        return {}

    mapping: dict[str, str] = {}
    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            image_id, scene = parts[0], normalize_scene(parts[1])
            mapping[image_id] = scene
    return mapping


def assign_scene(
    image_id: str,
    sequence_id: str | None = None,
    scene_map: dict[str, str] | None = None,
) -> str:
    """为单张图像分配场景标签。

    优先级:
        1. 显式 scene_map 中记录的 image_id；
        2. scene_map 中记录的 sequence_id；
        3. 兜底规则（按序列 ID 哈希取模），保证可运行。

    Args:
        image_id: 图像 ID 或文件名（不含扩展名）。
        sequence_id: 所属序列 ID。
        scene_map: 外部场景映射。

    Returns:
        场景英文 key。
    """
    scene_map = scene_map or {}

    if image_id in scene_map:
        return normalize_scene(scene_map[image_id])
    if sequence_id is not None and sequence_id in scene_map:
        return normalize_scene(scene_map[sequence_id])

    # 兜底：稳定哈希取模
    key = sequence_id if sequence_id else image_id
    return SCENES[hash(key) % FALLBACK_MOD]


def group_by_scene(
    image_ids: Iterable[str],
    scene_map: dict[str, str] | None = None,
    sequence_of: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """将图像列表按场景分组。

    Args:
        image_ids: 图像 ID 列表。
        scene_map: 外部场景映射。
        sequence_of: 图像 ID → 序列 ID 的映射（可选）。

    Returns:
        {scene: [image_id, ...]}。
    """
    sequence_of = sequence_of or {}
    groups: dict[str, list[str]] = {s: [] for s in SCENES}

    for image_id in image_ids:
        seq_id = sequence_of.get(image_id)
        scene = assign_scene(image_id, seq_id, scene_map)
        if scene not in groups:
            groups[scene] = []
        groups[scene].append(image_id)

    return groups


def available_scene_file(data_dir: str | os.PathLike) -> Path | None:
    """探测数据目录下可能存在的天气标注文件。

    Args:
        data_dir: 数据集根目录。

    Returns:
        若存在常见命名的标注文件则返回其路径，否则返回 None。
    """
    data_path = Path(data_dir)
    candidates = [
        "weather.txt",
        "scene_labels.txt",
        "scenes.txt",
        "weather_annotations.txt",
        "labels_weather.txt",
    ]
    for name in candidates:
        p = data_path / name
        if p.exists():
            return p
    return None