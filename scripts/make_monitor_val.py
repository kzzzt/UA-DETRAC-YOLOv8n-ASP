"""生成「训练期监控用」的 val 子集清单。

**为什么需要**：UA-DETRAC val 集有 16645 张图，ultralytics 每轮都在完整 val 上
验证，实测单轮验证耗时（约 240 s）甚至超过单轮训练（约 160 s），是服务器上最大的
时间开销。而训练期的验证只用于监控曲线与 best.pt 选择，与最终汇报的指标无关。

因此本脚本按「天气场景分层」抽取一个固定子集（默认每场景 750 张，共 3000 张），
写成 `<dataset_root>/monitor_val.txt`，配合 ``configs/ua_detrac_monitor.yaml``
作为训练时的 val 使用。**论文中所有指标仍由 ``scripts/eval_scenes.py`` 在完整
val 集上评估得到**（并只报告最后一个 epoch 的权重，避免 best.pt 选择偏差）。

抽样是确定性的（按名字排序后等间隔取），保证所有变体、所有种子用完全相同的子集。

用法:
    python scripts/make_monitor_val.py                 # 每场景 750 张
    python scripts/make_monitor_val.py --per-scene 500
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成训练期监控用的 val 子集清单")
    p.add_argument("--per-scene", type=int, default=750,
                   help="每个天气场景抽取的图像数（默认 750，共 4*750=3000 张）")
    p.add_argument("--out", default=None, help="输出清单路径（默认 <数据集根>/monitor_val.txt）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = dataset_root()
    val_dir = root / "images" / "val"
    scene_csv = root / "scene_annotations.csv"
    if not scene_csv.exists():
        print(f"[错误] 未找到场景标注: {scene_csv}\n       请先运行 scripts/classify_weather.py")
        return
    if not val_dir.is_dir():
        print(f"[错误] 未找到 val 目录: {val_dir}")
        return

    scene_of: dict[str, str] = {}
    with scene_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s = (row.get("scene") or "").strip().lower()
            if s in SCENES:
                scene_of[Path(row["image"]).name] = s

    per_scene: dict[str, list[Path]] = {s: [] for s in SCENES}
    for img in sorted(val_dir.glob("*.jpg")):
        s = scene_of.get(img.name)
        if s:
            per_scene[s].append(img)

    picked: list[Path] = []
    for s in SCENES:
        imgs = per_scene[s]
        n = min(args.per_scene, len(imgs))
        if n == 0:
            print(f"[警告] 场景 {s} 无图像")
            continue
        # 等间隔抽样：确定性、且能覆盖整个序列范围（连续帧高度相关，等间隔比随机更均匀）
        step = len(imgs) / n
        sel = [imgs[int(i * step)] for i in range(n)]
        picked.extend(sel)
        print(f"[场景] {s:<7}: 全量 {len(imgs):>6} 张 -> 抽 {len(sel):>5} 张")

    out = Path(args.out) if args.out else root / "monitor_val.txt"
    out.write_text("\n".join(str(p.resolve()) for p in picked) + "\n", encoding="utf-8")
    print(f"\n[输出] 监控子集清单: {out}（共 {len(picked)} 张，占 val 的 "
          f"{100 * len(picked) / max(sum(len(v) for v in per_scene.values()), 1):.1f}%）")
    print("[提醒] 论文指标请用 scripts/eval_scenes.py 在完整 val 集上评估。")


if __name__ == "__main__":
    main()
