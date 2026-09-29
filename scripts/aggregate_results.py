"""汇总多种子实验结果，输出「均值 ± 标准差」与论文用表格。

数据来源:
    1) 各运行的 ``weights/best.pt`` / ``weights/last.pt`` 中的 train_metrics（整体指标）；
       —— 同时报 best 与 last，避免"只报 best.pt 的取峰值偏差"。
    2) ``eval_scenes.py`` 产出的分场景 CSV（按运行名索引）。

用法:
    # 先对新增的多种子运行做分场景评估
    python scripts/eval_scenes.py --models baseline baseline_s1 baseline_s2 simam simam_s1 ... \
        --out_csv "C:\\Users\\康智童\\OneDrive\\桌面\\yolo\\runs\\eval\\summary_all.csv"

    # 再汇总
    python scripts/aggregate_results.py \
        --scene_csv "C:\\Users\\康智童\\OneDrive\\桌面\\yolo\\runs\\eval\\summary_all.csv"
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics as st
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ORDER = ["baseline", "simam", "p2", "eiou", "simam_p2", "asp"]
SCENES = ["sunny", "cloudy", "night", "rainy"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多种子结果汇总")
    p.add_argument("--runs_root",
                   default=r"E:\资料\基座\学习兴趣组\runs\detect\runs\train")
    p.add_argument("--scene_csv", default=None,
                   help="eval_scenes.py 产出的分场景 CSV")
    p.add_argument("--out_csv", default=None, help="汇总结果输出 CSV（可选）")
    p.add_argument("--significance", action="store_true",
                   help="额外输出 Welch t 检验（对照 baseline），论文主表可直接引用")
    p.add_argument("--ref", default="baseline", help="对照组变体名（默认 baseline）")
    return p.parse_args()


def welch_p(a: list[float], b: list[float]) -> float | None:
    """Welch t 检验 p 值；缺 scipy 或样本不足时返回 None。"""
    try:
        from scipy import stats as sps
    except Exception:  # noqa: BLE001
        return None
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return None
    try:
        return float(sps.ttest_ind(a, b, equal_var=False).pvalue)
    except Exception:  # noqa: BLE001
        return None


def sig_mark(p: float | None) -> str:
    if p is None:
        return ""
    if p < 0.01:
        return " **"
    if p < 0.05:
        return " *"
    return " n.s."


def split_name(name: str) -> tuple[str, int]:
    m = re.match(r"^(?P<var>.+?)(?:_s(?P<seed>\d+))?$", name)
    return m.group("var"), int(m.group("seed") or 0)


def read_ckpt_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tm = ckpt.get("train_metrics") or {}

    def g(k):
        v = tm.get(k)
        return float(v) if isinstance(v, (int, float)) else None

    return {
        "mAP50": g("metrics/mAP50(B)"),
        "mAP50-95": g("metrics/mAP50-95(B)"),
        "P": g("metrics/precision(B)"),
        "R": g("metrics/recall(B)"),
    }


def load_scene_csv(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[row["model"]] = row
    return out


def ms(vals: list[float]) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    if len(vals) == 1:
        return "%.4f" % vals[0]
    return "%.4f ± %.4f" % (st.mean(vals), st.stdev(vals))


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    if not runs_root.is_dir():
        print(f"[错误] 找不到 runs 根目录: {runs_root}")
        return

    scene = load_scene_csv(Path(args.scene_csv)) if args.scene_csv else {}

    # 收集每个运行的指标
    groups: dict[str, list[dict]] = {}
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir() or not (d / "weights" / "best.pt").exists():
            continue
        var, seed = split_name(d.name)
        rec = {"run": d.name, "seed": seed}
        for tag, fname in (("best", "best.pt"), ("last", "last.pt")):
            m = read_ckpt_metrics(d / "weights" / fname)
            if m:
                for k, v in m.items():
                    rec[f"{tag}_{k}"] = v
        if d.name in scene:
            row = scene[d.name]
            for s in SCENES:
                for metric in ("mAP50", "mAP50-95"):
                    key = f"{s}_{metric}"
                    if key in row:
                        try:
                            rec[key] = float(row[key])
                        except ValueError:
                            pass
            # eval_scenes.py 在**完整 val 集**上的整体指标优先。
            # 训练期若使用监控子集（configs/ua_detrac_monitor.yaml），checkpoint 里的
            # train_metrics 只反映子集；即便用完整 val 集，事后复评与训练期验证器的
            # 设置也略有差异。因此 mAP、P、R 一律以该 CSV 为准，保证本文件与
            # significance_report.md 同源（曾因只覆盖 mAP、漏掉 P/R 而在第四位小数上不一致）。
            for metric in ("mAP50", "mAP50-95", "P", "R"):
                key = f"overall_{metric}"
                if row.get(key):
                    try:
                        rec[f"best_{metric}"] = float(row[key])
                    except ValueError:
                        pass
        groups.setdefault(var, []).append(rec)

    if not groups:
        print("[错误] 没有找到任何含 best.pt 的运行。")
        return

    order = [v for v in ORDER if v in groups] + [v for v in groups if v not in ORDER]

    def table(title: str, cols: list[tuple[str, str]], fmt: str = "%.4f") -> None:
        print(f"\n### {title}\n")
        header = "| 变体 | n | " + " | ".join(c[0] for c in cols) + " |"
        sep = "|" + "---|" * (len(cols) + 2)
        print(header)
        print(sep)
        for v in order:
            recs = groups[v]
            cells = []
            for _, key in cols:
                vals = [r.get(key) for r in recs if r.get(key) is not None]
                cells.append(ms(vals))
            print(f"| {v} | {len(recs)} | " + " | ".join(cells) + " |")

    print("=" * 90)
    print("多种子结果汇总（best.pt 与 last.pt 分别给出，括号内为样本数 n）")
    print("=" * 90)

    table("整体指标（best.pt，完整 val 集评估）", [
        ("mAP50", "best_mAP50"), ("mAP50-95", "best_mAP50-95"),
        ("P", "best_P"), ("R", "best_R"),
    ])
    print("\n[说明] 上表四个指标均取自 eval_scenes.py 在**完整 val 集**上的评估，"
          "与 significance_report.md 同源。")
    table("整体指标（last.pt，checkpoint 训练期记录）", [
        ("mAP50", "last_mAP50"), ("mAP50-95", "last_mAP50-95"),
    ])
    print("\n[说明] last 权重未在完整 val 集上复评，上表保留 checkpoint 训练期数值。")

    if scene:
        table("分场景 mAP50（best.pt）",
              [(s, f"{s}_mAP50") for s in SCENES])
        table("分场景 mAP50-95（best.pt）",
              [(s, f"{s}_mAP50-95") for s in SCENES])
    else:
        print("\n[提示] 未提供 --scene_csv，跳过表格；如需分场景请先运行 eval_scenes.py")

    # 显著性检验（对照 --ref，默认 baseline）
    if args.significance:
        base_recs = groups.get(args.ref, [])
        if len(base_recs) < 2:
            print(f"\n[警告] 对照组 {args.ref} 样本不足（n={len(base_recs)}），跳过显著性检验")
        else:
            print("\n" + "=" * 90)
            print(f"显著性检验（Welch t 检验，对照 {args.ref}；"
                  f"* p<0.05, ** p<0.01, 其余为 n.s.）")
            print("=" * 90)
            sig_cols = [
                ("整体 mAP50（last.pt）", "last_mAP50"),
                ("整体 mAP50-95（last.pt）", "last_mAP50-95"),
                ("整体 mAP50（best.pt）", "best_mAP50"),
                ("整体 mAP50-95（best.pt）", "best_mAP50-95"),
            ] + [(f"{s} mAP50", f"{s}_mAP50") for s in SCENES] \
              + [(f"{s} mAP50-95", f"{s}_mAP50-95") for s in SCENES]
            for title, key in sig_cols:
                base = [r.get(key) for r in base_recs if r.get(key) is not None]
                if len(base) < 2:
                    continue
                bm = st.mean(base)
                print(f"\n### {title}\n")
                print(f"| 变体 | n | 值 (mean ± std) | 与 {args.ref} 差异 |")
                print("|---|---|---|---|")
                for v in order:
                    vals = [r.get(key) for r in groups[v] if r.get(key) is not None]
                    if not vals:
                        continue
                    if v == args.ref:
                        print(f"| {v} | {len(vals)} | {ms(vals)} | — |")
                        continue
                    diff = st.mean(vals) - bm
                    p = welch_p(vals, base)
                    mark = sig_mark(p) if p is not None else ""
                    print(f"| {v} | {len(vals)} | {ms(vals)} | {diff:+.4f}{mark} |")

    # 明细（便于核对与画图）
    print("\n### 逐次运行明细\n")
    keys = ["run", "seed", "best_mAP50", "best_mAP50-95", "last_mAP50", "last_mAP50-95"] + \
           [f"{s}_mAP50" for s in SCENES if any(f"{s}_mAP50" in r for r in sum(groups.values(), []))]
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for v in order:
        for r in sorted(groups[v], key=lambda x: x["seed"]):
            cells = []
            for k in keys:
                val = r.get(k)
                cells.append("%.4f" % val if isinstance(val, float) else str(val if val is not None else "—"))
            print("| " + " | ".join(cells) + " |")

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        all_keys = ["run", "seed"] + [k for k in keys if k not in ("run", "seed")]
        with out.open("w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            wr.writeheader()
            for v in order:
                for r in sorted(groups[v], key=lambda x: x["seed"]):
                    wr.writerow(r)
        print(f"\n[输出] 逐次运行明细已保存: {out}")


if __name__ == "__main__":
    main()
