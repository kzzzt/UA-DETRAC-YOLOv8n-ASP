"""UA-DETRAC 图像天气/光照场景标注脚本。

UA-DETRAC 官方未提供逐帧天气标签，但序列级可结合官方文档与
场景命名规则推断（如 MVI_20011 等原序列拍摄于白天晴/阴等）。

本脚本提供一条可运行、可解释的基线流程：
    1. 若存在人工/官方场景标注文件（CSV），优先使用。
    2. 否则基于图像亮度统计做粗粒度启发式划分：
         - 平均亮度高 -> 晴天/白天
         - 平均亮度低 -> 夜间
         - 饱和度低 -> 阴天/雨天（结合亮度判断）
    3. 输出 CSV（每帧一行），并支持按场景分组。

注意：启发式仅为占位 baseline，论文中“恶劣天气”结论应基于更可靠
的场景标注（官方序列描述或人工清洗）。本脚本输出产物可被
eval.py 的 --scene_file 直接读入。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import dataset_root  # noqa: E402
from utils.weather_annotations import normalize_scene, SCENES, SCENE_CN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 UA-DETRAC 天气/光照场景标注")
    parser.add_argument("--image_dir", default=str(dataset_root() / "images" / "val"),
                        help="待标注图像目录（通常为 val 或 test）")
    parser.add_argument("--output_csv", default=str(dataset_root() / "scene_annotations.csv"),
                        help="输出 CSV 路径")
    parser.add_argument("--existing_scene", default=None,
                        help="已有人工场景 CSV，优先读取")
    arg = parser.parse_args()
    return arg


def compute_brightness_saturation(img: Image.Image) -> tuple[float, float]:
    """计算图像平均亮度与平均饱和度。"""
    arr = np.asarray(img.convert("HSV"), dtype=np.float32)
    v_mean = arr[..., 2].mean()
    s_mean = arr[..., 1].mean()
    return float(v_mean), float(s_mean)


def heuristic_scene(brightness: float, saturation: float) -> str:
    """基于亮度/饱和度的粗粒度场景判断。

    Returns:
        scene: 'sunny' / 'cloudy' / 'rainy' / 'night' 之一。
    """
    if brightness < 45:
        return "night"
    if saturation < 55:
        return "rainy" if brightness < 90 else "cloudy"
    return "sunny"


def load_existing_scene(csv_path: Path) -> dict[str, str]:
    """读取已有场景 CSV，返回 {image_name: scene}。"""
    mapping: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = Path(row["image"]).name
            scene = normalize_scene(row["scene"])
            if scene:
                mapping[name] = scene
    return mapping


def load_sequence_weather(csv_path: Path) -> dict[str, str]:
    """读取 convert_detrac.py 生成的官方天气标签 {sequence: scene}。"""
    mapping: dict[str, str] = {}
    if not csv_path.exists():
        return mapping
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seq = (row.get("sequence") or "").strip()
            scene = normalize_scene(row.get("scene") or "")
            if seq and scene in SCENES:
                mapping[seq] = scene
    return mapping


def sequence_of_image(name: str) -> str | None:
    """从转换后的文件名（MVI_XXXXX_imgNNNNN.jpg）提取序列名（MVI_XXXXX）。"""
    parts = name.split("_")
    if len(parts) >= 2 and parts[0] == "MVI":
        return "_".join(parts[:2])  # MVI_XXXXX
    return None


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_csv = Path(args.output_csv)

    existing: dict[str, str] = {}
    if args.existing_scene and Path(args.existing_scene).exists():
        existing = load_existing_scene(Path(args.existing_scene))
        print(f"[读取] 已有人工场景标注 {len(existing)} 条。")

    if not image_dir.is_dir():
        print(f"[错误] 图像目录不存在: {image_dir}")
        return

    img_files = sorted(
        [p for p in image_dir.glob("*")
         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )
    if not img_files:
        print("[错误] 图像目录内未找到图像。")
        return

    official = load_sequence_weather(dataset_root() / "sequence_weather.csv")
    if official:
        print(f"[官方天气] 已读取 {len(official)} 个序列的官方天气标签，优先于启发式。")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    stats = {s: 0 for s in SCENES}

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "scene", "scene_cn", "brightness", "saturation"])
        for img_path in img_files:
            name = img_path.name
            if name in existing:
                scene = existing[name]
                brightness = saturation = -1.0
            else:
                seq = sequence_of_image(name)
                if seq in official:
                    scene = official[seq]
                    brightness = saturation = -1.0
                else:
                    img = Image.open(img_path)
                    brightness, saturation = compute_brightness_saturation(img)
                    scene = normalize_scene(heuristic_scene(brightness, saturation))
                    if scene not in SCENES:
                        scene = "sunny"
            stats[scene] = stats.get(scene, 0) + 1
            writer.writerow([name, scene, SCENE_CN.get(scene, scene),
                             f"{brightness:.2f}", f"{saturation:.2f}"])

    print("\n[完成] 场景划分统计:")
    for s in SCENES:
        print(f"  {SCENE_CN.get(s, s)} ({s}): {stats.get(s, 0)}")
    print(f"\n输出文件: {output_csv}")
    print("提示: 优先使用官方 sence_weather 标签；无官方标签时才回退到亮度启发式。")


if __name__ == "__main__":
    main()