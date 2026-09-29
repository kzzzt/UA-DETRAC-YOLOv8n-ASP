"""UA-DETRAC 标注格式 → YOLO 格式转换脚本。

UA-DETRAC 官方标注为逐序列 XML（每帧一个 <frame>，内含 <target>），
车辆类型属性 vehicle_type 取值为:
    car  / bus / van / others

本脚本将 car/bus/van 三类映射为 YOLO 类别下标 0/1/2，
忽略 others；box 由左上角 (left, top) 与宽高 (width, height)
转为 YOLO 归一化 xywh 中心格式。

输入目录约定（解压后常见结构）:
    DETRAC-train-data/
        Insight-MVT_Annotation_Train/
            MVI_20011/            # 序列帧图像目录
                img00001.jpg
                ...
    DETRAC-Train-Annotations-XML/
        MVI_20011.xml             # 对应序列标注

输出目录:
    <output_dir>/images/train  &  labels/train
    <output_dir>/images/val    &  labels/val
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Iterable

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import dataset_root  # noqa: E402

# 类别映射：UA-DETRAC vehicle_type -> YOLO 类别 id
CLASS_MAP = {
    "car": 0,
    "bus": 1,
    "van": 2,
}

# 验证集划分：官方 DETRAC 测试集序列号（此处保留占位，可覆盖）
# 说明：UA-DETRAC 训练集 60 序列 / 测试集 40 序列，
#       实际划分需根据官方发布清单设定。默认按序列名哈希均分。
DEFAULT_VAL_RATIO = 0.2


def stable_hash(text: str) -> int:
    """返回跨进程稳定的 32 位哈希。

    内置 hash() 受 PYTHONHASHSEED 随机化影响，同一序列名在不同进程中会得到
    不同哈希，导致验证集划分不稳定、消融实验之间不可比；改用 CRC32 保证可复现。
    """
    return zlib.crc32(text.encode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UA-DETRAC xml -> YOLO txt")
    parser.add_argument("--detrac_dir", default="./data",
                        help="DETRAC 数据根目录（含 DETRAC-Train-Annotations-XML 等）")
    parser.add_argument("--output_dir", default=str(dataset_root()),
                        help="YOLO 格式输出目录（默认 <datasets_dir>/ua_detrac）")
    parser.add_argument("--anno_dir", default=None,
                        help="XML 标注目录（默认自动探测）")
    parser.add_argument("--image_dir", default=None,
                        help="图像根目录（默认自动探测）")
    parser.add_argument("--frame_interval", type=int, default=1,
                        help="抽帧间隔，默认 1（每帧都用）")
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO,
                        help="按序列划分验证集比例")
    parser.add_argument("--val_sequences", default=None,
                        help="显式指定验证集序列列表文件（每行一个序列名）")
    return parser.parse_args()


def detect_dirs(detrac_dir: Path) -> tuple[Path, Path]:
    """自动探测 XML 标注目录与图像目录。"""
    xml_candidates = [
        detrac_dir / "DETRAC-Train-Annotations-XML",
        detrac_dir / "DETRAC-Test-Annotations-XML",
        detrac_dir / "annotations",
        detrac_dir / "xml",
    ]
    img_candidates = [
        detrac_dir / "DETRAC-train-data",
        detrac_dir / "DETRAC-test-data",
        detrac_dir / "images",
        detrac_dir / "Insight-MVT_Annotation_Train",
    ]

    anno_dir = next((p for p in xml_candidates if p.is_dir()), None)
    image_dir = next((p for p in img_candidates if p.is_dir()), None)

    if anno_dir is None:
        raise FileNotFoundError(
            f"未找到 XML 标注目录，请用 --anno_dir 指定。已检查: {xml_candidates}"
        )
    if image_dir is None:
        raise FileNotFoundError(
            f"未找到图像目录，请用 --image_dir 指定。已检查: {img_candidates}"
        )
    return anno_dir, image_dir


def find_image_files(image_root: Path, sequence: str) -> list[Path]:
    """查找某序列下的图像文件（支持嵌套结构，且只取本序列）。

    DETRAC 常见结构为 <root>/Insight-MVT_Annotation_Train/<sequence>/img*.jpg。
    优先精确匹配 <root>/<sequence>，否则在根下递归查找同名序列目录，
    避免误取其他序列的图像（否则不同序列的 img00001.jpg 会互相串扰）。
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    seq_dir = image_root / sequence
    if not seq_dir.is_dir():
        matches = [p for p in image_root.rglob(sequence) if p.is_dir()]
        seq_dir = matches[0] if matches else None
    if seq_dir is None:
        return []
    files = sorted(
        [p for p in seq_dir.glob("*") if p.suffix.lower() in exts]
    )
    if not files:
        # 递归查找（序列目录内）
        files = sorted(
            [p for p in seq_dir.rglob("*") if p.suffix.lower() in exts]
        )
    return files


def parse_xml(anno_path: Path) -> dict[int, list[tuple[str, float, float, float, float]]]:
    """解析单个序列 XML。

    Returns:
        {frame_num: [(class_name, x, y, w, h), ...]}，box 为像素坐标。
    """
    tree = ET.parse(anno_path)
    root = tree.getroot()
    frames: dict[int, list[tuple[str, float, float, float, float]]] = {}

    for frame in root.iter("frame"):
        num = int(frame.attrib.get("num", -1))
        targets: list[tuple[str, float, float, float, float]] = []
        for target in frame.iter("target"):
            box = target.find("box")
            attr = target.find("attribute")
            if box is None:
                continue
            vehicle_type = box.attrib.get("vehicle_type") or (
                attr.attrib.get("vehicle_type") if attr is not None else None
            )
            if vehicle_type is None or vehicle_type not in CLASS_MAP:
                continue
            left = float(box.attrib["left"])
            top = float(box.attrib["top"])
            width = float(box.attrib["width"])
            height = float(box.attrib["height"])
            targets.append((vehicle_type, left, top, width, height))
        frames[num] = targets
    return frames


def parse_weather(anno_path: Path) -> str:
    """读取序列官方天气标签。

    UA-DETRAC 官方 XML 的 <sequence_attribute> 带有 ``sence_weather`` 属性
    （官方即拼作 sence_weather），取值为 sunny/cloudy/night/rainy。
    """
    try:
        root = ET.parse(anno_path).getroot()
        attr = root.find("sequence_attribute")
        if attr is None:
            return ""
        return (attr.get("sence_weather") or attr.get("scene_weather") or "").strip().lower()
    except Exception:
        return ""


def to_yolo_box(
    left: float, top: float, width: float, height: float,
    img_w: int, img_h: int,
) -> tuple[float, float, float, float]:
    """像素 box → YOLO 归一化 xywh 中心格式。"""
    x_center = left + width / 2.0
    y_center = top + height / 2.0
    return (
        x_center / img_w,
        y_center / img_h,
        width / img_w,
        height / img_h,
    )


def write_label(
    label_path: Path,
    targets: Iterable[tuple[str, float, float, float, float]],
    img_w: int,
    img_h: int,
) -> None:
    """写入单张图像的 YOLO 标签。"""
    lines: list[str] = []
    for cls_name, left, top, w, h in targets:
        cls_id = CLASS_MAP[cls_name]
        xc, yc, wn, hn = to_yolo_box(left, top, w, h, img_w, img_h)
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_val_sequences(path: str | None) -> set[str] | None:
    """加载显式验证集序列列表。"""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"验证集序列列表不存在: {p}")
    return {
        line.strip() for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def main() -> None:
    args = parse_args()
    detrac_dir = Path(args.detrac_dir)
    output_dir = Path(args.output_dir)

    anno_dir = Path(args.anno_dir) if args.anno_dir else None
    image_dir = Path(args.image_dir) if args.image_dir else None
    if anno_dir is None or image_dir is None:
        a, i = detect_dirs(detrac_dir)
        anno_dir = anno_dir or a
        image_dir = image_dir or i

    print(f"[标注目录] {anno_dir}")
    print(f"[图像目录] {image_dir}")

    xml_files = sorted(anno_dir.glob("*.xml"))
    if not xml_files:
        print("[错误] 未找到任何 XML 标注文件。")
        return

    val_sequences = load_val_sequences(args.val_sequences)

    split_count = {"train": 0, "val": 0}
    split_img = {"train": 0, "val": 0}

    for xml_path in xml_files:
        sequence = xml_path.stem
        frames = parse_xml(xml_path)

        # 判断归属 train/val
        if val_sequences is not None:
            split = "val" if sequence in val_sequences else "train"
        else:
            # 稳定哈希按比例划分（内置 hash() 受 PYTHONHASHSEED 影响，改用 CRC32）
            split = "val" if (stable_hash(sequence) % 1000) < args.val_ratio * 1000 else "train"
        split_count[split] += 1

        # 定位图像
        image_files = find_image_files(image_dir, sequence)
        if not image_files:
            print(f"[警告] 序列 {sequence} 未找到图像，跳过。")
            continue

        # 帧号基座：DETRAC XML 的 <frame num> 不同版本可能从 0 或 1 起，
        # 取最小帧号作为偏移，使「按文件名排序后的第 k 张图像」正确对应。
        frame_offset = min((k for k in frames if k >= 0), default=0)

        for orig_idx, img_path in enumerate(image_files):
            if orig_idx % args.frame_interval != 0:
                continue
            img_name = img_path.stem
            frame_num = orig_idx + frame_offset

            targets = frames.get(frame_num, [])
            if not targets:
                continue

            # 图像实际尺寸（PIL 仅读文件头，开销极小），避免硬编码 960x540
            with Image.open(img_path) as im:
                img_w, img_h = im.size

            # 文件名加序列前缀，避免不同序列同名帧（img00001.jpg）互相覆盖
            out_name = f"{sequence}_{img_path.name}"
            out_img = output_dir / "images" / split / out_name
            out_lbl = output_dir / "labels" / split / (f"{sequence}_{img_name}.txt")
            out_img.parent.mkdir(parents=True, exist_ok=True)

            # 复制图像（符号链接在 Windows 不总是可用，使用硬拷贝）
            if not out_img.exists():
                shutil.copy2(img_path, out_img)

            write_label(out_lbl, targets, img_w, img_h)
            split_img[split] += 1

    print("\n[完成] 转换结果:")
    print(f"  序列划分: train={split_count['train']}, val={split_count['val']}")
    print(f"  图像数量: train={split_img['train']}, val={split_img['val']}")

    # 写出官方天气标签（若 XML 提供 sence_weather），供 classify_weather.py 使用
    weather_rows = [(xml.stem, parse_weather(xml)) for xml in xml_files]
    weather_rows = [(seq, w) for seq, w in weather_rows if w]
    if weather_rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        wcsv = output_dir / "sequence_weather.csv"
        with wcsv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sequence", "scene"])
            writer.writerows(weather_rows)
        print(f"[天气] 已写出官方天气标签 {len(weather_rows)} 条 -> {wcsv}")


if __name__ == "__main__":
    main()