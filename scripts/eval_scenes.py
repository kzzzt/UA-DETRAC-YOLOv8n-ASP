"""统一评估：对多个变体的 best.pt 做「整体 + 分天气场景」评估，生成论文用总表。

思路：把 val 集按官方天气标注（sunny/cloudy/night/rainy）拆成 4 份图像清单，
用 ultralytics 官方 val 分别评估（保证指标口径与训练时一致），最后汇总成表。

前置：需先运行 scripts/classify_weather.py 生成 scene_annotations.csv。

用法:
    python scripts/eval_scenes.py
    python scripts/eval_scenes.py --models baseline asp --out_csv runs/eval/test.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.paths import dataset_root  # noqa: E402
from utils.weather_annotations import SCENES  # noqa: E402

MODELS = ["baseline", "simam", "p2", "eiou", "simam_p2", "asp"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="整体 + 分天气场景统一评估")
    p.add_argument("--runs_root", default=r"E:\资料\基座\学习兴趣组\runs\detect\runs\train",
                   help="各变体训练输出根目录（含 <name>/weights/best.pt）")
    p.add_argument("--data", default="configs/ua_detrac.yaml", help="数据集 YAML")
    p.add_argument("--models", nargs="+", default=MODELS, help="待评估的变体名")
    p.add_argument("--weights-file", default="best.pt", choices=["best.pt", "last.pt"],
                   help="评估哪个权重：best.pt（默认）或 last.pt。"
                        "用「固定轮数训练」的服务器实验建议评估 last.pt，避免取峰值偏差")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=8, help="DataLoader 进程数（受限环境可用 0）")
    p.add_argument("--device", default=None)
    p.add_argument("--out_csv", default="runs/eval/summary_all.csv")
    return p.parse_args()


def load_scene_map(csv_path: Path) -> dict[str, str]:
    """读取 scene_annotations.csv -> {image_name: scene}。"""
    mapping: dict[str, str] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scene = (row.get("scene") or "").strip().lower()
            if scene in SCENES:
                mapping[Path(row["image"]).name] = scene
    return mapping


def build_scene_lists(scene_map: dict[str, str], val_dir: Path,
                      out_dir: Path) -> dict[str, Path]:
    """按场景写出图像清单 txt（绝对路径），返回 {scene: txt_path}。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_scene: dict[str, list[Path]] = {s: [] for s in SCENES}
    unmatched = 0
    for img in sorted(val_dir.glob("*.jpg")):
        scene = scene_map.get(img.name)
        if scene is None:
            unmatched += 1
            continue
        per_scene[scene].append(img)
    paths: dict[str, Path] = {}
    for s, imgs in per_scene.items():
        p = out_dir / f"{s}.txt"
        p.write_text("\n".join(str(i.resolve()) for i in imgs) + "\n", encoding="utf-8")
        paths[s] = p
        print(f"[场景] {s:<7}: {len(imgs):>6} 张 -> {p.name}")
    if unmatched:
        print(f"[警告] 有 {unmatched} 张 val 图像无场景标注，已跳过")
    return paths


def build_scene_yaml(scene_txt: Path, data_yaml: Path, out_dir: Path, scene: str) -> Path:
    """基于原数据集 YAML 生成「val 指向某场景清单」的临时 YAML。"""
    import yaml
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    d["val"] = str(scene_txt.resolve())
    p = out_dir / f"_scene_{scene}.yaml"
    p.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def main() -> None:
    args = parse_args()
    # noinspection PyUnresolvedReferences
    import models  # noqa: F401  注册自定义模块
    from ultralytics import YOLO

    root = dataset_root()
    val_dir = root / "images" / "val"
    scene_csv = root / "scene_annotations.csv"
    if not scene_csv.exists():
        print(f"[错误] 未找到场景标注: {scene_csv}\n       请先运行: python scripts/classify_weather.py")
        return
    if not val_dir.is_dir():
        print(f"[错误] 未找到 val 图像目录: {val_dir}")
        return

    # --data 兼容相对路径：先按当前目录找，找不到则按项目根解析（可从任意目录运行）
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        data_yaml = PROJECT_ROOT / data_yaml
    if not data_yaml.exists():
        print(f"[错误] 未找到数据集 YAML: {args.data}")
        return
    print(f"[数据配置] {data_yaml}")

    scene_map = load_scene_map(scene_csv)
    print(f"[场景] 载入标注 {len(scene_map)} 条")
    # 临时清单/临时 yaml 写到项目内（避免向数据集目录写文件）
    tmp_dir = PROJECT_ROOT / "runs" / "eval" / "scene_eval"
    scene_txt = build_scene_lists(scene_map, val_dir, tmp_dir)
    scene_yamls = {s: build_scene_yaml(t, data_yaml, tmp_dir, s)
                   for s, t in scene_txt.items()}

    runs_root = Path(args.runs_root)
    rows: list[dict] = []
    for name in args.models:
        w = runs_root / name / "weights" / args.weights_file
        if not w.exists():
            print(f"[跳过] {name}: 未找到 {w}")
            continue
        print(f"\n===== {name} ({args.weights_file}) =====")
        model = YOLO(str(w))
        common = dict(imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                      device=args.device, verbose=False,
                      project=str(PROJECT_ROOT / "runs" / "eval"), exist_ok=True)
        d = model.val(data=str(data_yaml), name=f"{name}_overall", **common).results_dict
        row = {
            "model": name,
            "overall_mAP50": round(float(d.get("metrics/mAP50(B)", 0.0)), 4),
            "overall_mAP50-95": round(float(d.get("metrics/mAP50-95(B)", 0.0)), 4),
            "overall_P": round(float(d.get("metrics/precision(B)", 0.0)), 4),
            "overall_R": round(float(d.get("metrics/recall(B)", 0.0)), 4),
        }
        print(f"  整体: mAP50={row['overall_mAP50']}  mAP50-95={row['overall_mAP50-95']}")
        for s in SCENES:
            ds = model.val(data=str(scene_yamls[s]), name=f"{name}_{s}", **common).results_dict
            row[f"{s}_mAP50"] = round(float(ds.get("metrics/mAP50(B)", 0.0)), 4)
            row[f"{s}_mAP50-95"] = round(float(ds.get("metrics/mAP50-95(B)", 0.0)), 4)
            print(f"  {s:<7}: mAP50={row[f'{s}_mAP50']}  mAP50-95={row[f'{s}_mAP50-95']}")
        rows.append(row)

    if not rows:
        print("[错误] 没有任何模型被成功评估。")
        return

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print("\n================ 整体 ================")
    print("%-10s %-9s %-11s %-8s %-8s" % ("model", "mAP50", "mAP50-95", "P", "R"))
    for r in rows:
        print("%-10s %-9.4f %-11.4f %-8.4f %-8.4f"
              % (r["model"], r["overall_mAP50"], r["overall_mAP50-95"], r["overall_P"], r["overall_R"]))

    for metric in ("mAP50", "mAP50-95"):
        print(f"\n================ 分场景 {metric} ================")
        print("%-10s %s" % ("model", " ".join("%-9s" % s for s in SCENES)))
        for r in rows:
            print("%-10s %s" % (r["model"], " ".join("%-9.4f" % r[f"{s}_{metric}"] for s in SCENES)))

    print(f"\n[输出] 完整结果已保存: {out}")


if __name__ == "__main__":
    main()
