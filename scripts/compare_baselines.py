"""基线对比脚本：YOLOv5n / YOLOv8n / YOLOv11n 在 UA-DETRAC 上的训练与评估。

对应《实验设计方案》3.1 节：通过三组 nano 基线的对比，论证选择 YOLOv8n
作为改进基线的合理性（精度与速度兼顾）。

用法:
    python scripts/compare_baselines.py \
        --models yolov5n yolov8n yolov11n \
        --data configs/ua_detrac.yaml --epochs 100 --imgsz 640

说明:
    - 默认加载各模型 COCO 预训练权重（yolov5n.pt / yolov8n.pt / yolov11n.pt）
      做迁移学习；ultralytics 会自动把检测头重建为数据集的 3 类（car/bus/van）。
    - 输出终端汇总表，并保存 runs/baselines/summary.csv。
    - 加 `--fps` 可粗测推理速度（对验证集前 32 张计时）。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import dataset_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多基线对比（YOLOv5n/v8n/v11n）")
    parser.add_argument("--models", nargs="+",
                        default=["yolov5n", "yolov8n", "yolov11n"],
                        help="待对比的模型名（对应 <name>.pt 预训练权重）")
    parser.add_argument("--data", default="configs/ua_detrac.yaml",
                        help="数据集 YAML")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None,
                        help="设备，留空自动选择（无 GPU 自动用 CPU）")
    parser.add_argument("--project", default="runs/baselines")
    parser.add_argument("--fps", action="store_true",
                        help="额外粗测推理速度（FPS，基于前 32 张验证图）")
    return parser.parse_args()


def measure_fps(model, val_dir: Path, imgsz: int, device: str, n: int = 32) -> float:
    """对验证集前 n 张图计时，返回每秒图像数（粗测）。"""
    imgs = sorted(
        [p for p in val_dir.glob("*")
         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )[:n]
    if not imgs:
        return float("nan")
    t0 = time.perf_counter()
    for p in imgs:
        model.predict(source=str(p), imgsz=imgsz, device=device, verbose=False)
    return len(imgs) / (time.perf_counter() - t0)


def main() -> None:
    args = parse_args()
    # noinspection PyUnresolvedReferences
    import models  # 注册自定义模块（基线虽不直接用到，保持环境一致）
    from ultralytics import YOLO

    rows: list[dict] = []
    val_dir = dataset_root() / "images" / "val"

    for name in args.models:
        weights = f"{name}.pt"
        print(f"\n===== [{name}] 训练 =====")
        model = YOLO(weights)
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            name=name,
            project=args.project,
            exist_ok=True,
        )

        print(f"\n===== [{name}] 评估 =====")
        metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
        d = metrics.results_dict
        n_params = sum(p.numel() for p in model.model.parameters())

        fps = measure_fps(model, val_dir, args.imgsz, args.device) if args.fps else float("nan")

        row = {
            "model": name,
            "params(M)": round(n_params / 1e6, 3),
            "mAP50": round(float(d.get("metrics/mAP50(B)", float("nan"))), 4),
            "mAP50-95": round(float(d.get("metrics/mAP50-95(B)", float("nan"))), 4),
            "precision": round(float(d.get("metrics/precision(B)", float("nan"))), 4),
            "recall": round(float(d.get("metrics/recall(B)", float("nan"))), 4),
            "FPS": round(fps, 1) if args.fps else "n/a",
        }
        rows.append(row)
        print(f"[{name}] params={row['params(M)']}M mAP50={row['mAP50']} "
              f"mAP50-95={row['mAP50-95']} FPS={row['FPS']}")

    # 汇总
    header = ["model", "params(M)", "mAP50", "mAP50-95", "precision", "recall", "FPS"]
    print("\n===== 基线对比汇总 =====")
    print("\t".join(header))
    for r in rows:
        print("\t".join(str(r[k]) for k in header))

    out_dir = Path(args.project)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[输出] 汇总表已保存: {out_csv}")


if __name__ == "__main__":
    main()
