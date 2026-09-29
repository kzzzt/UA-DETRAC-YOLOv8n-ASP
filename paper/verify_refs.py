"""用 Crossref 逐条核验 refs.bib 的真实性与字段正确性。

为什么需要这个脚本：手工整理文献时，标题/作者/年份容易核实，但**卷号与页码极易出错**
（本项目的 refs.bib 就曾被查出 3 处：一篇标题系误记、两处页码/卷号有误）。
本脚本把 refs.bib 里每一条拿去 Crossref（出版商官方登记的元数据）比对，一次性给出结论。

用法（需要能访问外网）:
    python verify_refs.py              # 只报告，不修改
    python verify_refs.py --fix        # 用 Crossref 的权威值修正 refs.bib（先自动备份）
    python verify_refs.py --only hou2021ca zhang2022eiou   # 只查指定条目

依赖: requests（或用标准库 urllib 回退，无需安装额外包）
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIB = HERE / "refs.bib"
MAILTO = "kangzt@stu2024.jnu.edu.cn"   # Crossref 要求提供联系方式（"polite pool"）
UA = f"refs-verifier/1.0 (mailto:{MAILTO})"

# 需要比对的字段：bibtex 字段名 -> Crossref 字段名
FIELD_MAP = {
    "title": "title",
    "year": "issued",
    "volume": "volume",
    "pages": "page",
    "journal": "container-title",
    "booktitle": "container-title",
}


def fetch(url: str, timeout: int = 30) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                print(f"    [网络错误] {type(e).__name__}: {e}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def parse_bib(text: str) -> list[dict]:
    """极简 bib 解析：按 @type{key, ... } 切块并抽取字段。"""
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        etype, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n", body + "\n", re.S):
            fields[fm.group(1).lower()] = re.sub(r"\s+", " ", fm.group(2)).strip()
        entries.append({"type": etype, "key": key, "fields": fields,
                        "span": (m.start(), m.end())})
    return entries


def norm(s: str) -> str:
    """归一化：去 LaTeX 花括号/命令、统一标点与大小写、压缩空白。"""
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = s.replace("{", "").replace("}", "").replace("--", "-")
    s = re.sub(r"[^0-9a-zA-Z]+", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def crossref_by_title(title: str, kind: str | None = None) -> dict | None:
    q = urllib.parse.quote(title)
    types = {"article": "journal-article", "inproceedings": "proceedings-article",
             "incollection": "book-chapter"}.get(kind or "", None)
    url = (f"https://api.crossref.org/works?query.bibliographic={q}&rows=5"
           f"&select=title,author,issued,container-title,page,volume,issue,DOI,type")
    if types:
        url += f"&filter=type:{types}"
    data = fetch(url)
    if not data:
        return None
    items = data.get("message", {}).get("items", [])
    target = norm(title)
    best, best_ratio = None, 0.0
    for it in items:
        cand = norm((it.get("title") or [""])[0])
        ratio = difflib.SequenceMatcher(None, target, cand).ratio()
        if ratio > best_ratio:
            best, best_ratio = it, ratio
    return best if best_ratio >= 0.80 else None


def crossref_by_doi(doi: str) -> dict | None:
    data = fetch(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    return data.get("message") if data else None


def cr_year(item: dict) -> str:
    try:
        return str(item["issued"]["date-parts"][0][0])
    except Exception:  # noqa: BLE001
        return ""


def cr_authors(item: dict) -> str:
    out = []
    for a in item.get("author", []) or []:
        fam = a.get("family", "")
        if fam:
            out.append(fam)
    return ", ".join(out[:3]) + (" et al." if len(out) > 3 else "")


def check_entry(e: dict, verbose: bool = True) -> dict:
    f = e["fields"]
    title = f.get("title", "")
    item = None
    if f.get("doi"):
        item = crossref_by_doi(f["doi"])
    if item is None and title:
        item = crossref_by_title(title, e["type"])
    res = {"key": e["key"], "title": title, "status": "?", "problems": [], "fixes": {}}
    if item is None:
        res["status"] = "未找到"
        res["problems"].append("Crossref 中未匹配到该标题 —— 需人工核实是否存在")
        return res

    cr_title = (item.get("title") or [""])[0]
    ratio = difflib.SequenceMatcher(None, norm(title), norm(cr_title)).ratio()
    if ratio < 0.90:
        res["problems"].append(f"标题不一致 -> {cr_title}")

    # 年份
    y = f.get("year", "")
    cy = cr_year(item)
    if y and cy and y != cy:
        res["problems"].append(f"年份 {y} -> {cy}")
        res["fixes"]["year"] = cy

    # 卷号 / 页码
    for bibf, crf in (("volume", "volume"), ("pages", "page")):
        cur = f.get(bibf, "")
        new = str(item.get(crf, "") or "").strip()
        if cur and new and norm(cur) != norm(new):
            res["problems"].append(f"{bibf} {cur} -> {new}")
            res["fixes"][bibf] = new
        elif not cur and new:
            res["fixes"].setdefault(bibf, new)

    res["status"] = "需修正" if res["problems"] else "一致"
    res["cr"] = {"title": cr_title, "year": cy, "authors": cr_authors(item),
                 "venue": (item.get("container-title") or [""])[0],
                 "volume": item.get("volume", ""), "issue": item.get("issue", ""),
                 "pages": item.get("page", ""), "doi": item.get("DOI", "")}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Crossref 文献核验")
    ap.add_argument("--fix", action="store_true", help="用 Crossref 权威值修正 refs.bib")
    ap.add_argument("--only", nargs="*", default=None, help="只检查这些 citation key")
    args = ap.parse_args()

    text = BIB.read_text(encoding="utf-8")
    entries = parse_bib(text)
    if args.only:
        entries = [e for e in entries if e["key"] in set(args.only)]
    print(f"[核验] {len(entries)} 条，数据源 Crossref（出版商登记元数据）\n")

    results = []
    for i, e in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] {e['key']}")
        r = check_entry(e)
        results.append(r)
        tag = {"一致": "OK  ", "需修正": "需修正", "未找到": "!! 未找到"}[r["status"]]
        print(f"    {tag}  {r['title'][:80]}")
        for p in r["problems"]:
            print(f"         · {p}")
        if r.get("cr"):
            c = r["cr"]
            print(f"         Crossref: {c['authors']} ({c['year']}) {c['venue']}"
                  f" {c['volume']}:{c['pages']}  doi:{c['doi']}")
        time.sleep(0.6)   # 对 Crossref 友好

    ok = sum(1 for r in results if r["status"] == "一致")
    fix = sum(1 for r in results if r["status"] == "需修正")
    miss = sum(1 for r in results if r["status"] == "未找到")
    print(f"\n=== 汇总：一致 {ok} | 需修正 {fix} | 未找到 {miss} ===")

    if miss:
        print("\n[注意] “未找到”的条目必须人工确认——投稿论文中出现无法核实的文献会被质疑。")
        for r in results:
            if r["status"] == "未找到":
                print(f"   · {r['key']}: {r['title'][:90]}")

    if args.fix and fix:
        backup = BIB.with_suffix(".bib.bak")
        shutil.copy2(BIB, backup)
        new_text = text
        for r in results:
            if not r["fixes"]:
                continue
            for bibf, val in r["fixes"].items():
                # 在该条目范围内替换字段值
                pat = re.compile(r"(@\w+\{" + re.escape(r["key"]) + r",.*?\n\})", re.S)
                def repl(m, bibf=bibf, val=val):
                    block = m.group(1)
                    if re.search(rf"\n\s*{bibf}\s*=\s*\{{", block):
                        return re.sub(rf"(\n\s*{bibf}\s*=\s*\{{)[^}}]*(\}})", rf"\g<1>{val}\g<2>", block)
                    return block.replace(",\n}", f",\n  {bibf} = {{{val}}}\n}}", 1)
                new_text = pat.sub(repl, new_text, count=1)
        BIB.write_text(new_text, encoding="utf-8")
        print(f"\n[已修正] {BIB.name}（备份：{backup.name}）——请重新检查并重跑本脚本确认")


if __name__ == "__main__":
    if sys.version_info < (3, 8):
        sys.exit("需要 Python 3.8+")
    main()
