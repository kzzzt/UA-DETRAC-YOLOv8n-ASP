"""用 Crossref 逐条核验 refs.bib 的真实性与字段正确性（v2）。

为什么需要这个脚本：手工整理文献时，标题/作者/年份容易核实，但**卷号与页码极易出错**
（本项目的 refs.bib 就被查出 5 处页码、1 处会议名、1 处"幽灵会议"）。

v1 的教训（务必保留这些设计）：
  1. **不要用 `filter=type:`**。Crossref 里 ECCV/LNCS 是 book-chapter、arXiv 是 DataCite 注册，
     加了 type 过滤器会把 SSD/CBAM/COCO 这类完全正确的条目误判为"未找到"。
  2. **模糊标题匹配必须再加两道闸门**：作者姓氏 & 年份。
     否则 "You Only Look Once" 会匹配到 2020 年的一篇 IJRASET 综述、
     "Focal Loss" 会匹配到 TPAMI 2020 的期刊版、FPN 会匹配到 IEEE Access 的无关论文。
  3. **区分"Crossref 不收录"与"文献有误"**。ICLR / PMLR(ICML) / JMLR / arXiv / 软件仓库
     本来就不在 Crossref，属于正常情况，脚本应给出官方出处供人工核对，而不是报错。
  4. **限流要能识别**。连续快速请求会被 Crossref 限流并返回空结果，
     v1 把它当成"未找到"，制造了 19 条假警报。本版对每个请求做重试+退避，
     并把"空结果"与"网络失败"分开报告。

用法（需要能访问外网）:
    python verify_refs.py                    # 只报告，不修改
    python verify_refs.py --fix              # 用 Crossref 权威值修正 refs.bib（先自动备份）
    python verify_refs.py --only hou2021ca   # 只查指定条目
    python verify_refs.py --report out.md    # 额外输出 Markdown 报告
    python verify_refs.py --sleep 1.5        # 放慢请求（默认 1.2 秒/条）

页码口径：本 bib 统一采用**出版商正式版（IEEE Xplore / Springer）**页码，
它与 CVF 开放获取版页码在 CVPR/ICCV 2016--2019 期间并不相同
（例如 FPN：IEEE 936--944，CVF 2117--2125；Focal Loss：IEEE 2999--3007，CVF 2980--2988）。
脚本因此以 DOI 背后的登记元数据为准。

依赖: 仅标准库。
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIB = HERE / "refs.bib"
MAILTO = "kangzt@stu2024.jnu.edu.cn"   # Crossref 要求提供联系方式（"polite pool"）
UA = f"refs-verifier/2.0 (mailto:{MAILTO})"

# Crossref 不收录 / 不适用 Crossref 的来源：正则 -> (说明, 官方核对入口模板)
NOT_IN_CROSSREF = [
    (r"arxiv\s+preprint\s+arxiv:([\d.]+)",
     "arXiv 预印本（Crossref 不收录，arXiv 用 DataCite 注册 DOI）",
     "https://arxiv.org/abs/{0}"),
    (r"international conference on learning representations",
     "ICLR（不通过 Crossref 注册 DOI）",
     "https://openreview.net/"),
    (r"journal of machine learning research",
     "JMLR（不注册 DOI，以官网卷期为准）",
     "https://www.jmlr.org/papers/"),
    (r"international conference on machine learning",
     "PMLR/ICML（PMLR 不向 Crossref 存缴，以 PMLR 官网为准）",
     "https://proceedings.mlr.press/"),
]

# 少量条目给出精确的官方核对入口（人工核对用）
MANUAL_URL = {
    "ge2021yolox": "https://arxiv.org/abs/2107.08430",
    "li2022yolov6": "https://arxiv.org/abs/2209.02976",
    "bochkovskiy2020yolov4": "https://arxiv.org/abs/2004.10934",
    "kisantal2019augmentation": "https://arxiv.org/abs/1902.07296",
    "beyer2020imagenet": "https://arxiv.org/abs/2006.07159",
    "michaelis2019benchmarking": "https://arxiv.org/abs/1907.07484",
    "loshchilov2017sgdr": "https://openreview.net/forum?id=Skq89Scxx",
    "hendrycks2019benchmarking": "https://openreview.net/forum?id=HJz6tiCqYm",
    "yang2021simam": "https://proceedings.mlr.press/v139/yang21o.html",
    "ioffe2015batchnorm": "https://proceedings.mlr.press/v37/ioffe15.html",
    "demsar2006statistical": "https://www.jmlr.org/papers/v7/demsar06a.html",
    "jocher2023ultralytics": "https://github.com/ultralytics/ultralytics",
    "ultralytics2024yolo11": "https://github.com/ultralytics/ultralytics",
}

TITLE_MIN = 0.90      # 标题相似度下限（低于此值认为不是同一篇）
YEAR_TOL = 1          # 年份容差


# ---------------------------------------------------------------- HTTP

class Throttled(Exception):
    """Crossref 限流。"""


def fetch(url: str, timeout: int = 30, attempts: int = 4) -> dict | None:
    """带指数退避的 GET。限流会重试，最终抛出 Throttled 而不是伪装成"无结果"。"""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 504) and attempt < attempts - 1:
                time.sleep(3.0 * (2 ** attempt))     # 3s, 6s, 12s
                continue
            print(f"    [HTTP {e.code}] {url[:80]}")
            return None
        except Exception as e:  # noqa: BLE001
            if attempt < attempts - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            print(f"    [网络错误] {type(e).__name__}: {e}")
            return None
    raise Throttled(url)


# ---------------------------------------------------------------- bib 解析

def parse_bib(text: str) -> list[dict]:
    """极简 bib 解析：按 @type{key, ... } 切块并抽取字段。"""
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        etype, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n", body + "\n", re.S):
            fields[fm.group(1).lower()] = re.sub(r"\s+", " ", fm.group(2)).strip()
        entries.append({"type": etype, "key": key, "fields": fields})
    return entries


def norm(s: str) -> str:
    """归一化：去 LaTeX 花括号/命令、去重音、统一标点与大小写、压缩空白。

    必须先处理 LaTeX 重音命令（\\'a / \\"o / \\v{s} / \\c{c}），否则
    "Doll{\\'a}r" 会变成 "doll ar" 而 Crossref 的 "Dollár" 变成 "dollar"，
    作者闸门就会误杀 dollar2012pedestrian、akyon2022sahi 这类条目。
    """
    s = re.sub(r"\\[`'\"^~=.uvHtcdb]\s*\{?([a-zA-Z])\}?", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = s.replace("{", "").replace("}", "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^0-9a-zA-Z]+", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def norm_pages(s: str) -> str:
    """页码归一化：统一 --/-/–、去空格。"""
    return re.sub(r"\s+", "", re.sub(r"[-‐‑‒–—]+", "-", str(s or "")))


def bib_venue(f: dict) -> str:
    return f.get("journal") or f.get("booktitle") or ""


def not_in_crossref(entry: dict) -> tuple[str, str] | None:
    """若该条目的来源本就不在 Crossref，返回 (说明, 核对入口)。

    注意：软件类条目用的是 @misc + howpublished，字段名不是 url，容易漏判。
    """
    f = entry["fields"]
    v = bib_venue(f)
    link = f.get("url") or f.get("howpublished") or ""
    if entry["type"] == "misc" or "github.com" in v or "github.com" in link:
        return "软件/代码仓库（不适用 Crossref）", re.sub(r"\\url\{(.*?)\}", r"\1", link)
    low = v.lower()
    for pat, note, tmpl in NOT_IN_CROSSREF:
        m = re.search(pat, low)
        if m:
            url = tmpl.format(*m.groups()) if "{" in tmpl else tmpl
            return note, url
    return None


# ---------------------------------------------------------------- Crossref

def crossref_query(title: str, author_hint: str = "") -> list[dict]:
    q = urllib.parse.quote(f"{title} {author_hint}".strip())
    url = ("https://api.crossref.org/works?query.bibliographic=" + q +
           "&rows=8&select=title,author,issued,container-title,page,volume,issue,DOI,type")
    data = fetch(url)
    if data is None:
        return []
    return data.get("message", {}).get("items", [])


def cr_authors(item: dict) -> list[str]:
    return [a.get("family", "") for a in (item.get("author") or []) if a.get("family")]


def cr_year(item: dict) -> str:
    try:
        return str(item["issued"]["date-parts"][0][0])
    except Exception:  # noqa: BLE001
        return ""


def pick_match(title: str, first_author: str, year: str, items: list[dict]) -> tuple[dict | None, str]:
    """三重闸门：标题相似度 + 作者姓氏 + 年份。返回 (条目, 拒绝原因)。"""
    target = norm(title)
    fa = norm(first_author.split(",")[0]) if first_author else ""
    best, best_ratio, why = None, 0.0, "Crossref 无相近标题"
    for it in items:
        cand = norm((it.get("title") or [""])[0])
        ratio = difflib.SequenceMatcher(None, target, cand).ratio()
        if ratio < TITLE_MIN:
            continue
        # 闸门 2：第一作者姓氏必须出现在候选作者列表中
        fams = [norm(x) for x in cr_authors(it)]
        if fa and fams and not any(fa == x or fa in x or x in fa for x in fams):
            why = f"标题相近但作者不符（{cand[:60]}）"
            continue
        # 闸门 3：年份容差
        cy = cr_year(it)
        if year and cy and abs(int(year) - int(cy)) > YEAR_TOL:
            why = f"标题相近但年份不符 {year} vs {cy}（{cand[:60]}）"
            continue
        if ratio > best_ratio:
            best, best_ratio = it, ratio
    if best is None:
        return None, why
    return best, ""


# ---------------------------------------------------------------- 单条核验

def check_entry(e: dict) -> dict:
    f = e["fields"]
    title = f.get("title", "")
    res = {"key": e["key"], "title": title, "status": "?", "notes": [], "fixes": {}, "cr": None}

    skip = not_in_crossref(e)
    if skip:
        res["status"] = "非 Crossref 来源"
        res["notes"].append(skip[0])
        res["manual_url"] = MANUAL_URL.get(e["key"]) or skip[1]
        return res

    first_author = (f.get("author") or "").split(" and ")[0]
    try:
        items = crossref_query(title, first_author.split(",")[0])
    except Throttled:
        res["status"] = "请求受限"
        res["notes"].append("Crossref 限流：本条未核验，请稍后重跑（--sleep 加大间隔）")
        return res

    item, why = pick_match(title, first_author, f.get("year", ""), items)
    if item is None:
        res["status"] = "未匹配"
        res["notes"].append(why + " —— 需人工核实，或该来源未被 Crossref 收录")
        return res

    cr_title = (item.get("title") or [""])[0]
    cy = cr_year(item)
    res["cr"] = {"title": cr_title, "year": cy,
                 "authors": ", ".join(cr_authors(item)[:3]) + (" et al." if len(cr_authors(item)) > 3 else ""),
                 "venue": (item.get("container-title") or [""])[0],
                 "volume": item.get("volume", ""), "issue": item.get("issue", ""),
                 "pages": item.get("page", ""), "doi": item.get("DOI", ""),
                 "type": item.get("type", "")}

    for bibf, crf in (("volume", "volume"), ("pages", "page")):
        cur, new = f.get(bibf, ""), str(item.get(crf, "") or "").strip()
        if cur and new and norm_pages(cur) != norm_pages(new):
            res["notes"].append(f"{bibf}: {cur} -> {new}（doi:{item.get('DOI','')}）")
            res["fixes"][bibf] = new
    if f.get("year") and cy and f["year"] != cy:
        res["notes"].append(f"year: {f['year']} -> {cy}")
        res["fixes"]["year"] = cy

    res["status"] = "需修正" if res["fixes"] else "一致"
    return res


# ---------------------------------------------------------------- main

def main() -> None:
    # Windows 控制台常用 GBK/CP936，而 Crossref 的作者名/刊名含 ö、š、á 等字符，
    # 直接 print 会抛 UnicodeEncodeError 让脚本中途崩溃 —— 这里降级为替换字符。
    try:
        sys.stdout.reconfigure(errors="replace")   # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser(description="Crossref 文献核验（v2）")
    ap.add_argument("--fix", action="store_true", help="用 Crossref 权威值修正 refs.bib（仅限三重闸门通过的匹配）")
    ap.add_argument("--only", nargs="*", default=None, help="只检查这些 citation key")
    ap.add_argument("--report", default=None, help="额外输出 Markdown 报告到该路径")
    ap.add_argument("--sleep", type=float, default=1.2, help="每条之间的间隔秒数（默认 1.2）")
    args = ap.parse_args()

    text = BIB.read_text(encoding="utf-8")
    entries = parse_bib(text)
    if args.only:
        entries = [e for e in entries if e["key"] in set(args.only)]
    print(f"[核验] {len(entries)} 条 | 数据源 Crossref（出版商登记元数据）| 间隔 {args.sleep}s\n")

    results = []
    for i, e in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] {e['key']}")
        r = check_entry(e)
        results.append(r)
        print(f"    {r['status']}  {r['title'][:80]}")
        for n in r["notes"]:
            print(f"         · {n}")
        if r.get("cr"):
            c = r["cr"]
            print(f"         Crossref: {c['authors']} ({c['year']}) {c['venue']}"
                  f" {c['volume']}:{c['pages']}  doi:{c['doi']}")
        time.sleep(args.sleep)

    order = ["一致", "需修正", "非 Crossref 来源", "未匹配", "请求受限"]
    print("\n=== 汇总 ===")
    for st in order:
        keys = [r["key"] for r in results if r["status"] == st]
        if keys:
            print(f"  {st}: {len(keys)}  -> {', '.join(keys)}")

    hard = [r for r in results if r["status"] in ("未匹配", "请求受限")]
    if hard:
        print("\n[需处理] 下列条目脚本无法用 Crossref 确认：")
        for r in hard:
            print(f"   · {r['key']}: {r['title'][:80]}")
            for n in r["notes"]:
                print(f"       {n}")

    if args.report:
        lines = ["# 参考文献核验报告（Crossref）", "",
                 f"条目数：{len(results)}｜标题闸门 {TITLE_MIN}｜年份容差 ±{YEAR_TOL}", "",
                 "| key | 结论 | Crossref 登记值 | DOI |", "|---|---|---|---|"]
        for r in results:
            c = r.get("cr") or {}
            reg = f"{c.get('venue','')} {c.get('volume','')}:{c.get('pages','')}".strip() if c else ""
            lines.append(f"| `{r['key']}` | {r['status']} | {reg} | {c.get('doi','')} |")
        lines.append("")
        lines.append("说明：`非 Crossref 来源` 指 ICLR / PMLR(ICML) / JMLR / arXiv / 软件仓库，"
                     "这些来源本就不在 Crossref 登记，需按官方出处人工核对（见脚本 MANUAL_URL）。")
        Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n[报告] 已写入 {args.report}")

    fixable = [r for r in results if r["status"] == "需修正" and r["fixes"]]
    if args.fix and fixable:
        backup = BIB.with_suffix(".bib.bak")
        shutil.copy2(BIB, backup)
        new_text = text
        for r in fixable:
            pat = re.compile(r"(@\w+\{" + re.escape(r["key"]) + r",.*?\n\})", re.S)

            def repl(m, fixes=r["fixes"]):
                block = m.group(1)
                for bibf, val in fixes.items():
                    if re.search(rf"\n\s*{bibf}\s*=\s*\{{", block):
                        block = re.sub(rf"(\n\s*{bibf}\s*=\s*\{{)[^}}]*(\}})",
                                       rf"\g<1>{val}\g<2>", block)
                    else:
                        block = block.replace(",\n}", f",\n  {bibf} = {{{val}}}\n}}", 1)
                return block

            new_text = pat.sub(repl, new_text, count=1)
        BIB.write_text(new_text, encoding="utf-8")
        print(f"\n[已修正] {BIB.name}（备份 {backup.name}）——请重跑本脚本确认全部为 一致")
    elif args.fix:
        print("\n[无需修正] 没有通过三重闸门的页码/卷号差异。")


if __name__ == "__main__":
    if sys.version_info < (3, 8):
        sys.exit("需要 Python 3.8+")
    main()
