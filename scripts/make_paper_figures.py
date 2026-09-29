"""从本地评估 CSV 生成论文插图（完全离线，不依赖服务器）。

输入（都是评估脚本产出的 CSV，本地即可复现）：
    --scratch_scene  runs/eval/summary_all.csv            从零训练 best.pt 的整体+分场景
    --scratch_size   runs/eval/ap_by_size_all.csv         从零训练 best.pt 的分尺寸
    --pre_scene      runs/eval_server/summary_pre_best.csv     预训练 best.pt
    --pre_last_scene runs/eval_server/summary_pre_last.csv     预训练 last.pt
    --pre_size       runs/eval_server/ap_by_size_pre_best.csv  预训练 best.pt 分尺寸

输出（默认写到 docs/figures/）：
    fig1_overall.png  整体 mAP50 / mAP50-95：从零(best) vs 预训练(best/last)，误差棒=标准差
    fig2_by_size.png  AP_small / AP_medium / AP_large：从零 vs 预训练
    fig3_scene.png    分场景 mAP50：四个场景 × 六变体（预训练 best）

用法:
    python scripts/make_paper_figures.py
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ORDER = ["baseline", "simam", "p2", "eiou", "simam_p2", "asp"]
SCENES = ["sunny", "cloudy", "night", "rainy"]
SCENE_CN = {"sunny": "Sunny", "cloudy": "Cloudy", "night": "Night", "rainy": "Rainy"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成论文插图（离线）")
    p.add_argument("--scratch_scene", default="../runs/eval/summary_all.csv")
    p.add_argument("--scratch_size", default="../runs/eval/ap_by_size_all.csv")
    p.add_argument("--pre_scene", default="../runs/eval_server/summary_pre_best.csv")
    p.add_argument("--pre_last_scene", default="../runs/eval_server/summary_pre_last.csv")
    p.add_argument("--pre_size", default="../runs/eval_server/ap_by_size_pre_best.csv")
    p.add_argument("--prefix", default="pre_", help="预训练运行名前缀")
    p.add_argument("--out_dir", default="docs/figures")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def load(path: Path, prefix: str = "") -> dict[str, list[dict]]:
    """读 CSV 并按变体分组（自动剥离前缀与 _s<seed>）。"""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, list[dict]] = {}
    for r in rows:
        name = r.get("model", "")
        if prefix and name.startswith(prefix):
            name = name[len(prefix):]
        var = re.match(r"^(?P<v>.+?)(?:_s\d+)?$", name).group("v")
        out.setdefault(var, []).append(r)
    return out


def stat(groups: dict[str, list[dict]], var: str, key: str) -> tuple[float, float]:
    vals = []
    for r in groups.get(var, []):
        v = r.get(key)
        if v not in (None, ""):
            try:
                vals.append(float(v))
            except ValueError:
                pass
    if not vals:
        return 0.0, 0.0
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def grouped_bars(ax, series: list[tuple[str, dict, str]], title: str, ylabel: str) -> None:
    import numpy as np
    n = len(ORDER)
    w = 0.8 / max(len(series), 1)
    x = np.arange(n)
    for i, (label, groups, key) in enumerate(series):
        means = [stat(groups, v, key)[0] for v in ORDER]
        stds = [stat(groups, v, key)[1] for v in ORDER]
        ax.bar(x + i * w - 0.4 + w / 2, means, w, yerr=stds, capsize=2.5, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)


def main() -> None:
    args = parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[错误] 缺少 matplotlib，无法生成插图")
        return

    base = PROJECT_ROOT
    s_scene = load((base / args.scratch_scene).resolve())
    s_size = load((base / args.scratch_size).resolve())
    p_scene = load((base / args.pre_scene).resolve(), args.prefix)
    p_last = load((base / args.pre_last_scene).resolve(), args.prefix)
    p_size = load((base / args.pre_size).resolve(), args.prefix)

    out_dir = (base / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出目录] {out_dir}")

    # ---- 图 1：整体 mAP50 / mAP50-95 ----
    if s_scene and p_scene:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        series50 = [("Scratch-100ep (best)", s_scene, "overall_mAP50"),
                    ("Pretrained-25ep (best)", p_scene, "overall_mAP50"),
                    ("Pretrained-25ep (last)", p_last, "overall_mAP50")]
        series95 = [("Scratch-100ep (best)", s_scene, "overall_mAP50-95"),
                    ("Pretrained-25ep (best)", p_scene, "overall_mAP50-95"),
                    ("Pretrained-25ep (last)", p_last, "overall_mAP50-95")]
        grouped_bars(axes[0], series50, "Overall mAP@0.5", "mAP50")
        grouped_bars(axes[1], series95, "Overall mAP@[0.5:0.95]", "mAP50-95")
        fig.suptitle("UA-DETRAC: variant comparison (mean ± std over 3 seeds)", fontsize=10)
        fig.tight_layout()
        p = out_dir / "fig1_overall.png"
        fig.savefig(p, dpi=args.dpi)
        plt.close(fig)
        print(f"[图1] {p}")

    # ---- 图 2：分目标尺寸 ----
    if s_size and p_size:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
        for ax, key, title in zip(axes, ("AP_small", "AP_medium", "AP_large"),
                                  ("AP small (<32²)", "AP medium (32²~96²)", "AP large (>96²)")):
            grouped_bars(ax, [("Scratch-100ep (best)", s_size, key),
                              ("Pretrained-25ep (best)", p_size, key)], title, key)
        fig.suptitle("COCO-style AP by object size (mean ± std over 3 seeds)", fontsize=10)
        fig.tight_layout()
        p = out_dir / "fig2_by_size.png"
        fig.savefig(p, dpi=args.dpi)
        plt.close(fig)
        print(f"[图2] {p}")

    # ---- 图 3：分场景 mAP50 ----
    if s_scene and p_scene:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, (groups, tag) in zip(axes, ((s_scene, "Scratch-100ep (best)"),
                                            (p_scene, "Pretrained-25ep (best)"))):
            series = [(SCENE_CN[s], groups, f"{s}_mAP50") for s in SCENES]
            grouped_bars(ax, series, f"{tag} — mAP50 by weather scene", "mAP50")
        fig.tight_layout()
        p = out_dir / "fig3_scene.png"
        fig.savefig(p, dpi=args.dpi)
        plt.close(fig)
        print(f"[图3] {p}")

    print("[完成] 插图已生成")


if __name__ == "__main__":
    main()
