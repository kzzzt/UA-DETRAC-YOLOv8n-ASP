"""UA-DETRAC 评估与分场景分析脚本。

功能:
    1. 在 val 集上计算整体 mAP（调用 ultralytics val）。
    2. 可选：按天气/光照场景（晴天/阴天/夜间/雨天）分别统计逐类 AP
       与 mAP，输出对比表格，直观展示恶劣条件下的性能。
    3. 可选：生成检测结果可视化图。

用法:
    # 整体评估
    python scripts/eval.py --weights runs/train/yolov8n-asp/weights/best.pt

    # 分场景评估（需先生成/指定场景标注）
    python scripts/eval.py --weights .../best.pt --scene_file data/ua_detrac/scene_annotations.csv

    # 可视化若干样例
    python scripts/eval.py --weights .../best.pt --visualize --num_vis 20

依赖:
    ultralytics>=8.0.0, pycocotools（ultralytics 自动安装）,
    numpy, pillow
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import compute_ap
from utils.paths import dataset_root
from utils.weather_annotations import (
    SCENES,
    SCENE_CN,
    normalize_scene,
)


def _image_size(path: Path) -> tuple[int, int]:
    """读取图像 (宽, 高)。PIL 仅解析文件头，不解码全图，开销极小。"""
    with Image.open(path) as im:
        return im.size  # (W, H)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UA-DETRAC 实验评估")
    parser.add_argument("--weights", required=True,
                        help="待评估模型权重路径（best.pt）")
    parser.add_argument("--data", default="configs/ua_detrac.yaml",
                        help="数据集 YAML 路径")
    parser.add_argument("--scene_file", default=None,
                        help="场景标注 CSV（image,scene,scene_cn,...）")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理图像尺寸")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="val 模式 NMS IoU")
    parser.add_argument("--device", default=None,
                        help="设备，留空自动选择（无 GPU 自动用 CPU）")
    parser.add_argument("--visualize", action="store_true",
                        help="生成可视化结果图")
    parser.add_argument("--num_vis", type=int, default=20,
                        help="可视化样本数量")
    parser.add_argument("--out_dir", default="runs/eval",
                        help="输出目录")
    return parser.parse_args()


def load_scene_csv(path: str | Path) -> dict[str, str]:
    """从 classify_weather.py 输出的 CSV 读取 {image_name: scene}。"""
    mapping: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = Path(row["image"]).name
            scene = normalize_scene(row["scene"])
            if scene in SCENES:
                mapping[name] = scene
    return mapping


def per_scene_ap(
    scenes: dict[str, str],
    names: dict[int, str],
    results: object,
    detections: list[dict],
) -> dict[str, dict[str, float]]:
    """按场景统计逐类 AP 与 mAP。

    这里复用 ultralytics val 得到的整体结果目录中的预测 JSON，
    但为保持零外部依赖，采用轻量自实现：基于 `detections`（
    每张图包含 box/scores/labels + 对应真值）分组计算。

    说明：若仅需整体 mAP，可直接看 ultralytics 输出。
    分场景 AP 使用本项目 utils.metrics 的贪心匹配实现。

    Args:
        scenes: {image_name: scene}。
        names: {class_id: class_name}。
        results: ultralytics 返回对象（未使用，保留接口）。
        detections: 列表，元素含 'image', 'pred_boxes', 'pred_scores',
                    'pred_labels', 'gt_boxes', 'gt_labels'。

    Returns:
        {scene: {'mAP50': float, 'per_class': {cls: ap}}}。
    """
    # 按 (scene, class) 汇总到各自类别桶
    per_img_bucket: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for det in detections:
        img = det["image"]
        scene = scenes.get(img)
        if scene is None:
            # 无场景标注的图像不参与任何场景统计（避免污染默认场景）
            continue
        # 每张图的所有预测需要按类别拆分
        for cid in set(det["pred_labels"]) | set(det["gt_labels"]):
            per_img_bucket[(scene, cid)].append(det)

    scene_ap: dict[str, dict[str, float]] = {s: {} for s in SCENES}
    per_class: dict[tuple[str, int], float] = {}

    for (scene, cid), dets in per_img_bucket.items():
        all_scores: list[float] = []
        all_matched: list[bool] = []
        n_gt_total = 0

        for det in dets:
            # 该图中该类的预测与真值
            mask_pred = [l == cid for l in det["pred_labels"]]
            mask_gt = [l == cid for l in det["gt_labels"]]
            pred_boxes = [b for b, m in zip(det["pred_boxes"], mask_pred) if m]
            pred_scores = [s for s, m in zip(det["pred_scores"], mask_pred) if m]
            gt_boxes = [b for b, m in zip(det["gt_boxes"], mask_gt) if m]
            n_gt_total += len(gt_boxes)

            if not pred_boxes:
                continue

            # 贪心匹配（单类）
            from utils.metrics import match_predictions
            tp, fp = match_predictions(
                pred_boxes, pred_scores, [cid] * len(pred_boxes),
                gt_boxes, [cid] * len(gt_boxes),
                iou_threshold=0.5,
            )
            for s, is_tp in zip(pred_scores, tp):
                all_scores.append(s)
                all_matched.append(is_tp)

        if n_gt_total == 0:
            per_class[(scene, cid)] = -1.0
            continue

        # 按分数降序计算 PR
        order = np.argsort(-np.asarray(all_scores))
        tp_cum = 0
        recalls: list[float] = []
        precisions: list[float] = []
        for idx in order:
            if all_matched[idx]:
                tp_cum += 1
            recalls.append(tp_cum / n_gt_total)
            precisions.append(tp_cum / (len(recalls)))
        per_class[(scene, cid)] = compute_ap(np.asarray(recalls), np.asarray(precisions))

    # 汇总
    for scene in SCENES:
        cids = sorted({c for (s, c) in per_class if s == scene})
        aps = [per_class[(scene, c)] for c in cids]
        valid = [a for a in aps if a >= 0]
        scene_ap[scene]["per_class"] = {}
        for c in cids:
            scene_ap[scene]["per_class"][names.get(c, str(c))] = per_class[(scene, c)]
        scene_ap[scene]["mAP50"] = float(np.mean(valid)) if valid else 0.0
        scene_ap[scene]["num_classes"] = len(cids)

    return scene_ap


def collect_detections(
    model,
    images: list[Path],
    conf: float,
    imgsz: int,
    device: str,
) -> tuple[list[dict], object]:
    """对图像集运行推理，并汇总每张图预测结果。

    Returns:
        (detections, results) —— detections 供分场景统计，
        results 为 ultralytics 整体结果。
    """
    detections: list[dict] = []
    for img_path in images:
        res = model.predict(
            source=str(img_path),
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            detections.append({
                "image": img_path.name,
                "pred_boxes": [], "pred_scores": [],
                "pred_labels": [], "gt_boxes": [], "gt_labels": [],
            })
            continue
        xyxy = boxes.xyxy.cpu().numpy().tolist()
        scores = boxes.conf.cpu().numpy().tolist()
        labels = boxes.cls.cpu().numpy().astype(int).tolist()
        detections.append({
            "image": img_path.name,
            "pred_boxes": xyxy,
            "pred_scores": scores,
            "pred_labels": labels,
            "gt_boxes": [], "gt_labels": [],
        })
    return detections, None


def main() -> None:
    args = parse_args()
    # noinspection PyUnresolvedReferences
    import models  # 注册自定义模块

    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"权重不存在: {weights}")

    print(f"[权重] {weights}")

    # 1. 整体评估
    model = YOLO(str(weights))
    print("[评估] 在 val 集上运行 ultralytics val ...")
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=args.visualize,
    )

    # 打印主要指标
    print("\n[整体指标]")
    for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"):
        val = metrics.results_dict.get(key, None)
        if val is not None:
            print(f"  {key}: {val:.4f}")

    # 2. 分场景评估
    if args.scene_file:
        scene_file = Path(args.scene_file)
        if scene_file.exists():
            scenes = load_scene_csv(scene_file)
            print(f"\n[场景] 加载场景标注 {len(scenes)} 条")

            # 获取验证图像路径与真值
            val_dir = dataset_root() / "images" / "val"
            label_dir = dataset_root() / "labels" / "val"
            imgs = sorted(
                [p for p in val_dir.glob("*")
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
            ) if val_dir.is_dir() else []

            detections, _ = collect_detections(model, imgs, args.conf, args.imgsz, args.device)

            # 读取真值标签到 detections
            names = model.names
            for det in detections:
                img_name = det["image"]
                label_path = label_dir / (Path(img_name).stem + ".txt")
                gt_boxes: list[list[float]] = []
                gt_labels: list[int] = []
                if label_path.exists():
                    # 真值框须与预测框同尺度：预测框为原图像素坐标，
                    # 故用图像实际尺寸还原归一化标签（不再硬编码 960x540）。
                    W, H = _image_size(val_dir / img_name)
                    for line in label_path.read_text(encoding="utf-8").splitlines():
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        cid = int(float(parts[0]))
                        xc, yc, w, h = map(float, parts[1:5])
                        x1 = (xc - w / 2) * W
                        y1 = (yc - h / 2) * H
                        x2 = (xc + w / 2) * W
                        y2 = (yc + h / 2) * H
                        gt_boxes.append([x1, y1, x2, y2])
                        gt_labels.append(cid)
                det["gt_boxes"] = gt_boxes
                det["gt_labels"] = gt_labels

            scene_ap = per_scene_ap(scenes, names, None, detections)

            print("\n[分场景 mAP50]")
            print(f"{'场景':<8}{'中文':<8}{'mAP50':>10}  类别明细")
            for s in SCENES:
                m = scene_ap[s]["mAP50"]
                line = f"{s:<8}{SCENE_CN.get(s, s):<8}{m:>10.4f}  "
                pc = scene_ap[s]["per_class"]
                detail = ", ".join(f"{k}={v:.3f}" for k, v in pc.items() if v >= 0)
                print(line + detail)

            # 写入 CSV
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / "per_scene_metrics.csv"
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["scene", "scene_cn", "mAP50"] + sorted(names.values()))
                for s in SCENES:
                    row = [s, SCENE_CN.get(s, s), f"{scene_ap[s]['mAP50']:.4f}"]
                    pc = scene_ap[s]["per_class"]
                    for cls in sorted(names.values()):
                        row.append(f"{pc.get(cls, -1):.4f}")
                    writer.writerow(row)
            print(f"\n[输出] 分场景指标已保存: {out_csv}")
        else:
            print(f"[警告] 场景标注文件不存在: {scene_file}")

    # 3. 可视化
    if args.visualize:
        print("\n[可视化] 生成检测结果图 ...")
        imgs = sorted(
            [p for p in (dataset_root() / "images" / "val").glob("*")
             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        )[: args.num_vis]
        for img_path in imgs:
            model.predict(
                source=str(img_path),
                conf=0.25,
                imgsz=args.imgsz,
                device=args.device,
                save=True,
                project=args.out_dir,
                name="visualize",
                exist_ok=True,
                verbose=False,
            )
        print(f"[输出] 可视化图像保存在 {Path(args.out_dir) / 'visualize'}")

    print("\n[完成] 评估流程结束。")


if __name__ == "__main__":
    main()