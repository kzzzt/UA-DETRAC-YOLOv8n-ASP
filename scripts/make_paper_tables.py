"""一键生成论文用表格（Markdown），把分散的评估 CSV 汇总成可直接粘贴的表格。

数据来源（都可选，缺哪个就跳过对应表）：

    从零训练（RTX 4060，100 轮）：
        --scratch_scene   runs/eval/summary_all.csv          eval_scenes.py 产出
        --scratch_size    runs/eval/ap_by_size_all.csv       eval_by_size.py 产出
        --scratch_runs    E:\\...\\runs\\detect\\runs\\train   用于读 best/last 的 checkpoint 指标

    预训练初始化（RTX 4090 D，25 轮，报告 last.pt）：
        --pre_scene       runs/eval_server/summary_pre_last.csv
        --pre_size        runs/eval_server/ap_by_size_pre_last.csv

用法:
    python scripts/make_paper_tables.py \
        --scratch_scene runs/eval/summary_all.csv \
        --scratch_size runs/eval/ap_by_size_all.csv \
        --pre_scene runs/eval_server/summary_pre_last.csv \
        --pre_size runs/eval_server/ap_by_size_pre_last.csv \
        --out docs/论文表格_自动生成.md
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成论文用表格")
    p.add_argument("--scratch_scene", default=None)
    p.add_argument("--scratch_size", default=None)
    p.add_argument("--scratch_runs", default=None,
                   help="从零训练的 runs 根目录（可选，用于 best/last checkpoint 指标）")
    p.add_argument("--pre_scene", default=None, help="预训练 best.pt 的分场景 CSV")
    p.add_argument("--pre_size", default=None, help="预训练 best.pt 的分尺寸 CSV")
    p.add_argument("--pre_last_scene", default=None, help="预训练 last.pt 的分场景 CSV")
    p.add_argument("--pre_last_size", default=None, help="预训练 last.pt 的分尺寸 CSV")
    p.add_argument("--prefix", default="pre_", help="预训练运行名前缀")
    p.add_argument("--patch-doc", default=None,
                   help="把预训练表写回该 Markdown 里的 AUTO:TABLE7A/7B 标记之间")
    p.add_argument("--out", default=None, help="输出 Markdown 路径（默认打印到终端）")
    return p.parse_args()


def welch_p(a: list[float], b: list[float]):
    try:
        from scipy import stats as sps
    except Exception:  # noqa: BLE001
        return None
    if len(a) < 2 or len(b) < 2:
        return None
    try:
        return float(sps.ttest_ind(a, b, equal_var=False).pvalue)
    except Exception:  # noqa: BLE001
        return None


def mark(p) -> str:
    if p is None:
        return ""
    return " **" if p < 0.01 else (" *" if p < 0.05 else " n.s.")


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def group(rows: list[dict], prefix: str = "") -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        name = r.get("model", "")
        if prefix and name.startswith(prefix):
            name = name[len(prefix):]
        var = re.match(r"^(?P<v>.+?)(?:_s\d+)?$", name).group("v")
        out.setdefault(var, []).append(r)
    return out


def stat_cell(groups: dict[str, list[dict]], var: str, key: str) -> str:
    vals = []
    for r in groups.get(var, []):
        v = r.get(key)
        if v not in (None, ""):
            try:
                vals.append(float(v))
            except ValueError:
                pass
    if not vals:
        return "—"
    if len(vals) == 1:
        return "%.4f" % vals[0]
    return "%.4f ± %.4f" % (st.mean(vals), st.stdev(vals))


def sig_table(groups: dict[str, list[dict]], title: str, key: str,
              ref: str = "baseline") -> list[str]:
    base = [float(r[key]) for r in groups.get(ref, []) if r.get(key) not in (None, "")]
    if len(base) < 2:
        return [f"\n### {title}\n", "（对照组样本不足，跳过）\n"]
    out = [f"\n### {title}\n",
           f"| 变体 | n | 值 (mean ± std) | 与 {ref} 差异 |",
           "|---|---|---|---|"]
    for v in ORDER:
        recs = groups.get(v, [])
        vals = [float(r[key]) for r in recs if r.get(key) not in (None, "")]
        if not vals:
            continue
        if v == ref:
            out.append(f"| {v} | {len(vals)} | {stat_cell(groups, v, key)} | — |")
            continue
        diff = st.mean(vals) - st.mean(base)
        out.append(f"| {v} | {len(vals)} | {stat_cell(groups, v, key)} | "
                   f"{diff:+.4f}{mark(welch_p(vals, base))} |")
    out.append("")
    return out


def ckpt_metrics(runs_root: Path) -> dict[str, dict[str, list[float]]]:
    """读每个运行的 best.pt / last.pt 训练期指标 -> {variant: {key: [values]}}"""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return {}
    data: dict[str, dict[str, list[float]]] = {}
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        var = re.match(r"^(?P<v>.+?)(?:_s\d+)?$", d.name).group("v")
        for tag, fname in (("best", "best.pt"), ("last", "last.pt")):
            p = d / "weights" / fname
            if not p.exists():
                continue
            try:
                tm = torch.load(p, map_location="cpu", weights_only=False).get("train_metrics") or {}
            except Exception:  # noqa: BLE001
                continue
            for src, dst in (("metrics/mAP50(B)", f"{tag}_mAP50"),
                             ("metrics/mAP50-95(B)", f"{tag}_mAP50-95")):
                v = tm.get(src)
                if isinstance(v, (int, float)):
                    data.setdefault(var, {}).setdefault(dst, []).append(float(v))
    return data


def ms_list(vals: list[float]) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return "%.4f" % vals[0]
    return "%.4f ± %.4f" % (st.mean(vals), st.stdev(vals))


def pretrained_table(scene_csv: Path, size_csv: Path | None, prefix: str) -> list[str]:
    """生成「预训练对比」表：mAP50 / mAP50-95（带 vs baseline 差异与显著性）+ 三个尺寸档。"""
    rows = load_rows(scene_csv)
    g = group(rows, prefix)
    size_g: dict[str, list[dict]] = {}
    if size_csv and size_csv.exists():
        size_g = group(load_rows(size_csv), prefix)

    base50 = [float(r["overall_mAP50"]) for r in g.get("baseline", []) if r.get("overall_mAP50")]
    base95 = [float(r["overall_mAP50-95"]) for r in g.get("baseline", []) if r.get("overall_mAP50-95")]

    out = ["| 变体 | n | mAP50 | mAP50-95 | AP_small | AP_medium | AP_large |",
           "|------|---|-------|----------|----------|-----------|----------|"]
    for v in ORDER:
        recs = g.get(v, [])
        if not recs:
            continue
        n = len(recs)
        c50 = stat_cell(g, v, "overall_mAP50")
        c95 = stat_cell(g, v, "overall_mAP50-95")
        if v != "baseline" and len(base50) >= 2:
            vals = [float(r["overall_mAP50"]) for r in recs if r.get("overall_mAP50")]
            if vals:
                c50 += " (%+.4f%s)" % (st.mean(vals) - st.mean(base50), mark(welch_p(vals, base50)))
            vals = [float(r["overall_mAP50-95"]) for r in recs if r.get("overall_mAP50-95")]
            if vals and len(base95) >= 2:
                c95 += " (%+.4f%s)" % (st.mean(vals) - st.mean(base95), mark(welch_p(vals, base95)))
        small = stat_cell(size_g, v, "AP_small") if size_g else "待填"
        med = stat_cell(size_g, v, "AP_medium") if size_g else "待填"
        large = stat_cell(size_g, v, "AP_large") if size_g else "待填"
        out.append(f"| {v} | {n} | {c50} | {c95} | {small} | {med} | {large} |")
    return out


def patch_doc(doc: Path, scene_best: Path, size_best: Path | None,
              scene_last: Path, size_last: Path | None, prefix: str) -> None:
    """把 7a/7b 两张表写回 Markdown 的 AUTO 标记之间。"""
    text = doc.read_text(encoding="utf-8")
    blocks = {
        "TABLE7A": pretrained_table(scene_best, size_best, prefix),
        "TABLE7B": pretrained_table(scene_last, size_last, prefix),
    }
    for tag, lines in blocks.items():
        begin, end = f"<!-- AUTO:{tag}:BEGIN -->", f"<!-- AUTO:{tag}:END -->"
        if begin not in text or end not in text:
            print(f"[警告] {doc.name} 缺少 {tag} 标记，跳过")
            continue
        head, rest = text.split(begin, 1)
        _old, tail = rest.split(end, 1)
        text = head + begin + "\n" + "\n".join(lines) + "\n" + end + tail
    doc.write_text(text, encoding="utf-8")
    print(f"[已写回] {doc}")


def main() -> None:
    args = parse_args()

    if args.patch_doc:
        doc = Path(args.patch_doc)
        if not doc.is_absolute():
            doc = PROJECT_ROOT / doc
        if not args.pre_scene or not args.pre_last_scene:
            print("[错误] --patch-doc 需要同时提供 --pre_scene 与 --pre_last_scene")
            return
        patch_doc(doc,
                  Path(args.pre_scene),
                  Path(args.pre_size) if args.pre_size else None,
                  Path(args.pre_last_scene),
                  Path(args.pre_last_size) if args.pre_last_size else None,
                  args.prefix)
        return

    L: list[str] = []
    L.append("# 论文表格（自动生成）\n")
    L.append("> 由 `scripts/make_paper_tables.py` 从评估 CSV 生成，可随时重跑复现。")
    L.append("> 显著性为 Welch t 检验（* p<0.05、** p<0.01、其余 n.s.）。\n")

    # ---------- 从零训练：主表（best/last，来自 checkpoint 训练期记录）----------
    if args.scratch_runs:
        rr = Path(args.scratch_runs)
        if rr.is_dir():
            ck = ckpt_metrics(rr)
            L.append("\n## 表 A 从零训练主表（100 轮，RTX 4060，n=3）\n")
            L.append("| 变体 | n | best mAP50 | best mAP50-95 | last mAP50 | last mAP50-95 |")
            L.append("|---|---|---|---|---|---|")
            for v in ORDER:
                d = ck.get(v, {})
                n = len(d.get("best_mAP50", [])) or len(d.get("last_mAP50", []))
                L.append(f"| {v} | {n} | {ms_list(d.get('best_mAP50', []))} | "
                         f"{ms_list(d.get('best_mAP50-95', []))} | "
                         f"{ms_list(d.get('last_mAP50', []))} | "
                         f"{ms_list(d.get('last_mAP50-95', []))} |")
            L.append("")

    # ---------- 从零训练：分场景 / 分尺寸 ----------
    if args.scratch_scene and Path(args.scratch_scene).exists():
        rows = load_rows(Path(args.scratch_scene))
        g = group(rows)
        L.append("\n## 表 B 从零训练分场景（best.pt，n=3）\n")
        for metric, label in (("mAP50", "mAP50"), ("mAP50-95", "mAP50-95")):
            L.append(f"\n### {label}\n")
            L.append("| 变体 | n | " + " | ".join(SCENES) + " |")
            L.append("|---" * (len(SCENES) + 2) + "|")
            for v in ORDER:
                if v not in g:
                    continue
                L.append(f"| {v} | {len(g[v])} | " +
                         " | ".join(stat_cell(g, v, f"{s}_{metric}") for s in SCENES) + " |")
        L.append("\n### 分场景显著性（对照 baseline）")
        for s in SCENES:
            L += sig_table(g, f"{s} mAP50", f"{s}_mAP50")
            L += sig_table(g, f"{s} mAP50-95", f"{s}_mAP50-95")

    if args.scratch_size and Path(args.scratch_size).exists():
        rows = load_rows(Path(args.scratch_size))
        g = group(rows)
        L.append("\n## 表 C 从零训练分目标尺寸 AP（COCO 口径，best.pt，n=3）\n")
        L.append("| 变体 | n | AP_small | AP_medium | AP_large |")
        L.append("|---|---|---|---|---|")
        for v in ORDER:
            if v not in g:
                continue
            L.append(f"| {v} | {len(g[v])} | {stat_cell(g, v, 'AP_small')} | "
                     f"{stat_cell(g, v, 'AP_medium')} | {stat_cell(g, v, 'AP_large')} |")
        for k in ("AP_small", "AP_medium", "AP_large"):
            L += sig_table(g, f"分尺寸 {k}", k)

    # ---------- 预训练对比 ----------
    if args.pre_scene and Path(args.pre_scene).exists():
        rows = load_rows(Path(args.pre_scene))
        g = group(rows, args.prefix)
        L.append(f"\n## 表 D 预训练初始化（25 轮，RTX 4090 D，last.pt，n=3）\n")
        L.append("| 变体 | n | mAP50 | mAP50-95 | P | R |")
        L.append("|---|---|---|---|---|---|")
        for v in ORDER:
            if v not in g:
                continue
            L.append(f"| {v} | {len(g[v])} | {stat_cell(g, v, 'overall_mAP50')} | "
                     f"{stat_cell(g, v, 'overall_mAP50-95')} | "
                     f"{stat_cell(g, v, 'overall_P')} | {stat_cell(g, v, 'overall_R')} |")
        L.append("\n### 整体显著性（对照 baseline）")
        L += sig_table(g, "整体 mAP50", "overall_mAP50")
        L += sig_table(g, "整体 mAP50-95", "overall_mAP50-95")
        L.append("\n### 分场景\n")
        L.append("| 变体 | n | sunny | cloudy | night | rainy |")
        L.append("|---|---|---|---|---|---|")
        for v in ORDER:
            if v not in g:
                continue
            L.append(f"| {v} | {len(g[v])} | " +
                     " | ".join(stat_cell(g, v, f"{s}_mAP50") for s in SCENES) + " |")
        for s in SCENES:
            L += sig_table(g, f"预训练 {s} mAP50", f"{s}_mAP50")

    if args.pre_size and Path(args.pre_size).exists():
        rows = load_rows(Path(args.pre_size))
        g = group(rows, args.prefix)
        L.append("\n## 表 E 预训练初始化分尺寸 AP（last.pt，n=3）\n")
        L.append("| 变体 | n | AP_small | AP_medium | AP_large |")
        L.append("|---|---|---|---|---|")
        for v in ORDER:
            if v not in g:
                continue
            L.append(f"| {v} | {len(g[v])} | {stat_cell(g, v, 'AP_small')} | "
                     f"{stat_cell(g, v, 'AP_medium')} | {stat_cell(g, v, 'AP_large')} |")
        for k in ("AP_small", "AP_medium", "AP_large"):
            L += sig_table(g, f"预训练分尺寸 {k}", k)

    text = "\n".join(L) + "\n"
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"[输出] {out}（{len(text.splitlines())} 行）")
    else:
        print(text)


if __name__ == "__main__":
    main()
