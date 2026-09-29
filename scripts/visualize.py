"""检测结果可视化与对比脚本。

功能:
    1. 对指定图像/目录运行推理并保存标注结果图。
    2. 支持多模型对比可视化（并排展示基线 vs 改进模型）。
    3. 可选：仅可视化真值框（用于数据检查）。

用法:
    # 单模型可视化
    python scripts/visualize.py --weights runs/train/yolov8n-asp/weights/best.pt \
        --source data/ua_detrac/images/val --num 20

    # 对比可视化（基线 vs 改进）
    python scripts/visualize.py --weights \
        runs/train/baseline/weights/best.pt \
        runs/train/yolov8n-asp/weights/best.pt \
        --source data/ua_detrac/images/val --num 20

    # 可视化真值框
    python scripts/visualize.py --source data/ua_detrac/images/val \
        --label_dir data/ua_detrac/labels/val --num 10 --gt_only

输出:
    runs/visualize/<timestamp>/ 下保存结果图。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检测结果可视化")
    parser.add_argument("--weights", nargs="+", default=None,
                        help="一个或多个模型权重路径（多模型则对比）")
    parser.add_argument("--source", required=True,
                        help="待可视化图像目录或单张图像")
    parser.add_argument("--label_dir", default=None,
                        help="真值标签目录（用于 gt_only 模式）")
    parser.add_argument("--gt_only", action="store_true",
                        help="仅可视化真值框")
    parser.add_argument("--num", type=int, default=20,
                        help="可视化样本数量")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理图像尺寸")
    parser.add_argument("--device", default=None,
                        help="设备，留空自动选择（无 GPU 自动用 CPU）")
    parser.add_argument("--out_dir", default="runs/visualize",
                        help="输出根目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.gt_only:
        # 真值可视化
        import cv2
        import numpy as np

        source = Path(args.source)
        label_dir = Path(args.label_dir) if args.label_dir else None
        if label_dir is None:
            raise ValueError("--gt_only 模式需要指定 --label_dir")

        img_files = _collect_images(source, args.num)
        out_dir = Path(args.out_dir) / f"{datetime.now():%Y%m%d_%H%M%S}_gt"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[真值可视化] 共 {len(img_files)} 张图像 -> {out_dir}")
        for img_path in img_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            label_path = label_dir / (img_path.stem + ".txt")
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cid = int(float(parts[0]))
                    xc, yc, w, h = map(float, parts[1:5])
                    x1 = int((xc - w / 2) * W)
                    y1 = int((yc - h / 2) * H)
                    x2 = int((xc + w / 2) * W)
                    y2 = int((yc + h / 2) * H)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"cls{cid}", (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imwrite(str(out_dir / img_path.name), img)
        print(f"[完成] 真值可视化结果保存在 {out_dir}")
        return

    if not args.weights:
        raise ValueError("请提供 --weights（至少一个模型权重）")

    # noinspection PyUnresolvedReferences
    import models  # 注册自定义模块
    from ultralytics import YOLO

    source = Path(args.source)
    img_files = _collect_images(source, args.num)

    if len(args.weights) == 1:
        _visualize_single(YOLO, args, img_files)
    else:
        _visualize_compare(YOLO, args, img_files)


def _collect_images(source: Path, num: int) -> list[Path]:
    """收集待可视化图像路径。"""
    if source.is_file():
        return [source]
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(
        [p for p in source.glob("*") if p.suffix.lower() in exts]
    )[:num]


def _visualize_single(YOLO, args: argparse.Namespace, img_files: list[Path]) -> None:
    """单模型可视化。"""
    model = YOLO(args.weights[0])
    out_path = Path(args.out_dir) / f"{datetime.now():%Y%m%d_%H%M%S}_single"
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[单模型可视化] {args.weights[0]} -> {out_path}")
    for img_path in img_files:
        model.predict(
            source=str(img_path),
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            save=True,
            project=str(out_path.parent),
            name=out_path.name,
            exist_ok=True,
            verbose=False,
        )
    print(f"[完成] 结果保存在 {out_path}")


def _visualize_compare(YOLO, args: argparse.Namespace, img_files: list[Path]) -> None:
    """多模型对比可视化，并排拼接结果。"""
    import cv2
    import numpy as np

    print("[对比可视化] 加载模型 ...")
    models = [YOLO(w) for w in args.weights]

    out_path = Path(args.out_dir) / f"{datetime.now():%Y%m%d_%H%M%S}_compare"
    out_path.mkdir(parents=True, exist_ok=True)
    tmp_root = out_path / "_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    for img_path in img_files:
        rendered: list[np.ndarray] = []
        for mi, model in enumerate(models):
            res = model.predict(
                source=str(img_path),
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            # 保存单图标注结果
            tmp_model_dir = tmp_root / f"model{mi}"
            tmp_model_dir.mkdir(parents=True, exist_ok=True)
            model.predict(
                source=str(img_path),
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                save=True,
                project=str(tmp_model_dir),
                name="out",
                exist_ok=True,
                verbose=False,
            )
            out_img = next((p for p in (tmp_model_dir / "out").glob("*")
                            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}), None)
            if out_img is not None:
                rendered.append(cv2.imread(str(out_img)))
            else:
                rendered.append(cv2.imread(str(img_path)))

        # 水平拼接
        if rendered:
            max_h = max(r.shape[0] for r in rendered)
            resized = []
            for r in rendered:
                h, w = r.shape[:2]
                scale = max_h / h
                resized.append(cv2.resize(r, (int(w * scale), max_h)))
            canvas = np.hstack(resized)
            cv2.imwrite(str(out_path / img_path.name), canvas)

    # 清理临时目录
    import shutil
    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"[完成] 对比可视化结果保存在 {out_path}")


if __name__ == "__main__":
    main()