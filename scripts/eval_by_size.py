"""按目标尺寸评估 AP_small / AP_medium / AP_large（ultralytics 原生 COCO 评估）。

**原理**：ultralytics 8.4 在 ``save_json=True`` 且数据集非 COCO 时，会自动：
    1. 把 val 集真值转成 COCO 格式（``gdict``）；
    2. 把预测转成 COCO 格式（``jdict``）；
    3. 调用 ``faster-coco-eval`` 跑标准 COCOeval，产出
       ``metrics/mAP_small(B)`` / ``metrics/mAP_medium(B)`` / ``metrics/mAP_large(B)``
       （均为 AP@[.5:.95]，COCO 标准面积档：small <32²、medium 32²~96²、large >96²）。

因此本脚本不做任何自实现匹配，**直接复用官方口径**，保证 ``all`` 列与
训练时的 mAP50/mAP50-95 完全一致。

依赖:
    pip install "faster-coco-eval>=1.6.7"     # 首次运行 ultralytics 也会自动尝试安装

用法:
    python scripts/eval_by_size.py                      # 默认 6 个主实验
    python scripts/eval_by_size.py --models baseline baseline_s1 baseline_s2 ...
    python scripts/eval_by_size.py --out_csv "C:\\...\\ap_by_size.csv"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MAIN_MODELS = ["baseline", "simam", "p2", "eiou", "simam_p2", "asp"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按目标尺寸评估 AP（ultralytics 原生 COCO 评估）")
    p.add_argument("--models", nargs="+", default=MAIN_MODELS)
    p.add_argument("--weights-file", default="best.pt", choices=["best.pt", "last.pt"],
                   help="评估哪个权重：best.pt（默认）或 last.pt；"
                        "固定轮数训练的服务器实验请用 last.pt")
    p.add_argument("--runs_root",
                   default=r"E:\资料\基座\学习兴趣组\runs\detect\runs\train")
    p.add_argument("--data", default="configs/ua_detrac.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=8, help="DataLoader 进程数（受限环境用 0）")
    p.add_argument("--device", default=None)
    p.add_argument("--out_csv", default="runs/eval/ap_by_size.csv")
    return p.parse_args()


def check_dependency() -> bool:
    try:
        import faster_coco_eval  # noqa: F401
        return True
    except ImportError:
        print('[错误] 缺少依赖 faster-coco-eval，请先安装：\n'
              '       pip install "faster-coco-eval>=1.6.7"\n'
              '       （ultralytics 运行时会自动尝试安装，但显式安装更可靠）')
        return False


def eval_one(weights: Path, data_yaml: Path, args) -> dict | None:
    """跑一次官方 val + COCO 评估，返回含 small/medium/large 的 stats。"""
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.detect import DetectionValidator

    overrides = dict(
        model=str(weights), data=str(data_yaml), task="detect", mode="val",
        save_json=True, plots=False, verbose=False, exist_ok=True,
        imgsz=args.imgsz, batch=args.batch, workers=args.workers,
        project=str(PROJECT_ROOT / "runs" / "eval"), name=f"size_{weights.parent.parent.name}",
    )
    if args.device:
        overrides["device"] = args.device

    model = YOLO(str(weights))
    validator = DetectionValidator(args=get_cfg(overrides=overrides))
    validator(model=model.model)

    stats = dict(validator.metrics.results_dict)
    # 触发原生 COCO 评估（填充 mAP_small/medium/large）
    stats = validator.eval_json(stats)
    return stats


def main() -> None:
    args = parse_args()
    if not check_dependency():
        return

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        data_yaml = PROJECT_ROOT / data_yaml
    if not data_yaml.exists():
        print(f"[错误] 未找到数据集 YAML: {args.data}")
        return

    rows = []
    n = len(args.models)
    for i, name in enumerate(args.models, 1):
        w = Path(args.runs_root) / name / "weights" / args.weights_file
        if not w.exists():
            print(f"[跳过] {name}: 未找到 {w}")
            continue
        print(f"\n===== ({i}/{n}) {name} =====")
        try:
            st = eval_one(w, data_yaml, args)
        except Exception as e:  # noqa: BLE001
            print(f"[失败] {name}: {type(e).__name__}: {e}")
            continue

        def g(k, default=0.0):
            v = st.get(k, default)
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        row = {
            "model": name,
            "mAP50": round(g("metrics/mAP50(B)"), 4),
            "mAP50-95": round(g("metrics/mAP50-95(B)"), 4),
            "AP_small": round(g("metrics/mAP_small(B)"), 4),
            "AP_medium": round(g("metrics/mAP_medium(B)"), 4),
            "AP_large": round(g("metrics/mAP_large(B)"), 4),
            "P": round(g("metrics/precision(B)"), 4),
            "R": round(g("metrics/recall(B)"), 4),
        }
        rows.append(row)
        print("  mAP50=%.4f  mAP50-95=%.4f | AP_small=%.4f  AP_medium=%.4f  AP_large=%.4f"
              % (row["mAP50"], row["mAP50-95"], row["AP_small"], row["AP_medium"], row["AP_large"]))

    if not rows:
        print("[错误] 没有任何模型被成功评估。")
        return

    out = Path(args.out_csv)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print("\n================ 汇总（COCO 口径，AP 均为 AP@[.5:.95]） ================")
    print("%-11s %-9s %-11s %-11s %-11s %-11s" %
          ("model", "mAP50", "mAP50-95", "AP_small", "AP_medium", "AP_large"))
    for r in rows:
        print("%-11s %-9.4f %-11.4f %-11.4f %-11.4f %-11.4f"
              % (r["model"], r["mAP50"], r["mAP50-95"], r["AP_small"], r["AP_medium"], r["AP_large"]))
    print(f"\n[输出] {out}")


if __name__ == "__main__":
    main()
