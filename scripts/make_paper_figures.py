"""从本地评估 CSV 生成论文插图（完全离线，不依赖服务器）。

**关于字号**：IEEE/Springer 双栏模板的正文宽度是 160mm ≈ 6.3in。如果按 11--13in 出图
再缩到 6.3in 排进版面，缩放系数只有 0.5 左右，图上 10pt 的字到版面上只剩 5pt，几乎无法阅读。
因此本脚本默认**按最终物理尺寸出图**（fig_width=6.3in），图上的 7--8pt 就是版面真实的 7--8pt。
如需给其他宽度排版，用 --fig_width 传入目标英寸宽度即可。

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

用法（任意装有 matplotlib 的 Python 均可）:
    python scripts/make_paper_figures.py
    python scripts/make_paper_figures.py --fig_width 3.15   # 单栏排版
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

BASE_FONT = 7.0          # 版面上的真实磅值：图按 1:1 尺寸生成，所见即所得


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成论文插图（离线）")
    p.add_argument("--scratch_scene", default="../runs/eval/summary_all.csv")
    p.add_argument("--scratch_size", default="../runs/eval/ap_by_size_all.csv")
    p.add_argument("--pre_scene", default="../runs/eval_server/summary_pre_best.csv")
    p.add_argument("--pre_last_scene", default="../runs/eval_server/summary_pre_last.csv")
    p.add_argument("--pre_size", default="../runs/eval_server/ap_by_size_pre_best.csv")
    p.add_argument("--prefix", default="pre_", help="预训练运行名前缀")
    p.add_argument("--out_dir", default="docs/figures")
    p.add_argument("--fig_width", type=float, default=6.3,
                   help="图片物理宽度（英寸）。160mm 双栏版面用 6.3，单栏用 3.15")
    p.add_argument("--dpi", type=int, default=400)
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


def apply_style(plt) -> None:
    """版面级字号：7pt 左右是双栏论文插图的常见下限，再小就难读。"""
    plt.rcParams.update({
        "font.size": BASE_FONT,
        "font.family": "sans-serif",
        "axes.titlesize": BASE_FONT + 0.5,
        "axes.labelsize": BASE_FONT,
        "xtick.labelsize": BASE_FONT - 0.5,
        "ytick.labelsize": BASE_FONT - 0.5,
        "legend.fontsize": BASE_FONT - 0.5,
        "figure.titlesize": BASE_FONT + 1.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "grid.linewidth": 0.4,
    })


def grouped_bars(ax, series: list[tuple[str, dict, str]], title: str, ylabel: str) -> None:
    import numpy as np
    n = len(ORDER)
    w = 0.8 / max(len(series), 1)
    x = np.arange(n)
    top = 0.0
    for i, (label, groups, key) in enumerate(series):
        means = [stat(groups, v, key)[0] for v in ORDER]
        stds = [stat(groups, v, key)[1] for v in ORDER]
        ax.bar(x + i * w - 0.4 + w / 2, means, w, yerr=stds, capsize=1.8,
               error_kw={"linewidth": 0.6, "capthick": 0.6}, label=label)
        top = max(top, max((m + s) for m, s in zip(means, stds)) if means else 0.0)
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, rotation=32, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, top * 1.12 if top > 0 else 1.0)
    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def shared_legend(fig, axes) -> None:
    """整图共用一个图例，避免图例压住柱子、也省出绘图区高度。"""
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        return
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
               frameon=False, handlelength=1.4, columnspacing=1.4,
               handletextpad=0.5, bbox_to_anchor=(0.5, -0.015))


def main() -> None:
    args = parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[错误] 缺少 matplotlib，无法生成插图")
        return

    apply_style(plt)
    fig_w = args.fig_width
    fig_h = fig_w / 2.5          # 宽高比 2.5，与版面观感一致

    base = PROJECT_ROOT
    s_scene = load((base / args.scratch_scene).resolve())
    s_size = load((base / args.scratch_size).resolve())
    p_scene = load((base / args.pre_scene).resolve(), args.prefix)
    p_last = load((base / args.pre_last_scene).resolve(), args.prefix)
    p_size = load((base / args.pre_size).resolve(), args.prefix)

    out_dir = (base / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出目录] {out_dir}  (图片宽度 {fig_w}in @ {args.dpi}dpi)")

    # ---- 图 1：整体 mAP50 / mAP50-95 ----
    if s_scene and p_scene:
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
        grouped_bars(axes[0], [("Scratch-100ep (best)", s_scene, "overall_mAP50"),
                               ("Pretrained-25ep (best)", p_scene, "overall_mAP50"),
                               ("Pretrained-25ep (last)", p_last, "overall_mAP50")],
                     "Overall mAP@0.5", "mAP")
        grouped_bars(axes[1], [("Scratch-100ep (best)", s_scene, "overall_mAP50-95"),
                               ("Pretrained-25ep (best)", p_scene, "overall_mAP50-95"),
                               ("Pretrained-25ep (last)", p_last, "overall_mAP50-95")],
                     "Overall mAP@[0.5:0.95]", "mAP")
        fig.suptitle("UA-DETRAC: variant comparison (mean $\\pm$ std over 3 seeds)")
        shared_legend(fig, axes)
        fig.tight_layout(rect=(0, 0.07, 1, 0.94))
        p = out_dir / "fig1_overall.png"
        fig.savefig(p, dpi=args.dpi)
        plt.close(fig)
        print(f"[图1] {p}")

    # ---- 图 2：分目标尺寸 ----
    if s_size and p_size:
        fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))
        for ax, key, title in zip(axes, ("AP_small", "AP_medium", "AP_large"),
                                  ("AP small (<32$^2$)", "AP medium (32$^2$--96$^2$)",
                                   "AP large (>96$^2$)")):
            grouped_bars(ax, [("Scratch-100ep (best)", s_size, key),
                              ("Pretrained-25ep (best)", p_size, key)], title, "AP")
        fig.suptitle("COCO-style AP by object size (mean $\\pm$ std over 3 seeds)")
        shared_legend(fig, axes)
        fig.tight_layout(rect=(0, 0.08, 1, 0.92))
        p = out_dir / "fig2_by_size.png"
        fig.savefig(p, dpi=args.dpi)
        plt.close(fig)
        print(f"[图2] {p}")

    # ---- 图 3：分场景 mAP50 ----
    if s_scene and p_scene:
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
        for ax, (groups, tag) in zip(axes, ((s_scene, "Scratch-100ep (best)"),
                                            (p_scene, "Pretrained-25ep (best)"))):
            series = [(SCENE_CN[s], groups, f"{s}_mAP50") for s in SCENES]
            grouped_bars(ax, series, f"{tag}: mAP@0.5 by scene", "mAP@0.5")
        shared_legend(fig, axes)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        p = out_dir / "fig3_scene.png"
        fig.savefig(p, dpi=args.dpi)
        plt.close(fig)
        print(f"[图3] {p}")

    print("[完成] 插图已生成")


if __name__ == "__main__":
    main()
