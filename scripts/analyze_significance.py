"""对多种子结果做均值±标准差与显著性检验（Welch t 检验，对照 baseline）。

用法:
    python scripts/analyze_significance.py \
        --scene_csv "C:\\Users\\康智童\\OneDrive\\桌面\\yolo\\runs\\eval\\summary_all.csv"

输出（Markdown）:
    整体 / 分场景指标，每格为 ``mean ± std``，与 baseline 的差异标注显著性：
        *   p < 0.05
        **  p < 0.01
        n.s. 不显著
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ORDER = ["baseline", "simam", "p2", "eiou", "simam_p2", "asp"]
SCENES = ["sunny", "cloudy", "night", "rainy"]

try:
    from scipy import stats as _sps  # 可选
except Exception:  # noqa: BLE001
    _sps = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多种子显著性分析")
    p.add_argument("--scene_csv", required=True, help="eval_scenes.py 产出的分场景 CSV")
    p.add_argument("--prefix", default="",
                   help="运行名前缀（如 pre_），用于把 pre_baseline_s1 归到 baseline 组")
    p.add_argument("--ref", default="baseline",
                   help="对照组变体名（默认 baseline）")
    p.add_argument("--runs_root",
                   default=r"E:\资料\基座\学习兴趣组\runs\detect\runs\train")
    return p.parse_args()


def split_name(name: str, prefix: str = "") -> str:
    if prefix and name.startswith(prefix):
        name = name[len(prefix):]
    return re.match(r"^(?P<var>.+?)(?:_s\d+)?$", name).group("var")


def welch_p(a: list[float], b: list[float]) -> float | None:
    """Welch t 检验 p 值；无 scipy 时返回 None。"""
    if _sps is None or len(a) < 2 or len(b) < 2:
        return None
    try:
        return float(_sps.ttest_ind(a, b, equal_var=False).pvalue)
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


def fmt(vals: list[float], diff: float | None, p: float | None) -> str:
    if not vals:
        return "—"
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    txt = "%.4f ± %.4f" % (m, s)
    if diff is not None and p is not None:
        txt += " (%+.4f%s)" % (diff, sig_mark(p))
    return txt


def main() -> None:
    args = parse_args()
    with Path(args.scene_csv).open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    data: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        var = split_name(r["model"], args.prefix)
        d = data.setdefault(var, {})
        for k, v in r.items():
            if k == "model" or v in ("", None):
                continue
            try:
                d.setdefault(k, []).append(float(v))
            except ValueError:
                pass

    if _sps is None:
        print("[提示] 未安装 scipy，无法给出 p 值（仅显示均值±标准差）\n")
    else:
        print("[提示] 显著性标记：* p<0.05, ** p<0.01, n.s. 不显著（Welch t 检验 vs baseline）\n")

    # 自动识别需要分析的数值列（排除 nGT 等计数列）
    skip_tokens = ("nGT", "n_gt")

    def is_numeric(k: str) -> bool:
        for r in rows:
            v = r.get(k)
            if v in ("", None):
                continue
            try:
                float(v)
            except ValueError:
                return False
        return True

    metrics = [(k, k) for k in rows[0].keys()
               if k != "model" and not any(t in k for t in skip_tokens) and is_numeric(k)]

    for title, key in metrics:
        base = data.get(args.ref, {}).get(key, [])
        print(f"### {title}\n")
        print(f"| 变体 | n | 值 (mean ± std) | 与 {args.ref} 差异 |")
        print("|---|---|---|---|")
        for var in ORDER:
            vals = data.get(var, {}).get(key, [])
            if not vals:
                continue
            diff = (st.mean(vals) - st.mean(base)) if base else None
            p = welch_p(vals, base) if var != args.ref and base else None
            if var == args.ref:
                print("| %s | %d | %s | — |" % (var, len(vals), fmt(vals, None, None)))
            else:
                d_txt = "%+.4f%s" % (diff, sig_mark(p)) if diff is not None else "—"
                print("| %s | %d | %s | %s |" % (var, len(vals), fmt(vals, None, None), d_txt))
        print()


if __name__ == "__main__":
    main()
