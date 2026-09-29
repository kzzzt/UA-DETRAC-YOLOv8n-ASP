"""校验 main.tex 的编译前风险：引用键、环境配对、括号、非 ASCII、图片存在性。"""
import re
from collections import Counter
from pathlib import Path

PAPER = Path(__file__).resolve().parent
tex = (PAPER / "main.tex").read_text(encoding="utf-8")
bib = (PAPER / "refs.bib").read_text(encoding="utf-8")

# 1) 括号与数学环境
stripped = tex.replace(r"\{", "").replace(r"\}", "")
print("brace balance:", stripped.count("{") - stripped.count("}"), "(0=OK)")
dollars = len(re.findall(r"(?<!\\)\$", tex))
print("unescaped $ count:", dollars, "->", "even OK" if dollars % 2 == 0 else "ODD! 检查")

# 2) 非 ASCII
bad = {}
for i, line in enumerate(tex.splitlines(), 1):
    for ch in line:
        if ord(ch) > 127:
            bad.setdefault(ch, []).append(i)
print("non-ascii:", {repr(c): v[:3] for c, v in bad.items()} or "none")

# 3) 引用键
cites = set()
for m in re.finditer(r"\\cite\{([^}]*)\}", tex):
    cites.update(k.strip() for k in m.group(1).split(",") if k.strip())
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
print(f"cited: {len(cites)} | bib entries: {len(keys)}")
missing = sorted(cites - keys)
print("MISSING in refs.bib:", missing or "none")
unused = sorted(keys - cites)
print("unused bib entries:", len(unused))

# 4) 环境配对
c1 = Counter(re.findall(r"\\begin\{(\w+\*?)\}", tex))
c2 = Counter(re.findall(r"\\end\{(\w+\*?)\}", tex))
mismatch = {k: (c1[k], c2[k]) for k in set(c1) | set(c2) if c1[k] != c2[k]}
print("env mismatch:", mismatch or "none")

# 5) 图片（支持 \graphicspath + 无扩展名写法）
gpaths = re.findall(r"\\graphicspath\{((?:\{[^}]*\})*)\}", tex)
dirs = re.findall(r"\{([^}]*)\}", gpaths[0]) if gpaths else ["./"]
print("graphicspath:", dirs or "none")
for m in re.finditer(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", tex):
    name = m.group(1)
    found = None
    for d in dirs or ["./"]:
        for ext in ("", ".png", ".pdf", ".jpg", ".eps"):
            cand = PAPER / d / f"{name}{ext}"
            if cand.exists():
                found = cand
                break
        if found:
            break
    print(f"figure: {name:16s} -> {found.relative_to(PAPER) if found else 'NOT FOUND'}")

# 6) 表格行/列一致性（booktabs 表格的 & 数量）
for m in re.finditer(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", tex, re.S):
    spec, body = m.group(1), m.group(2)
    ncol = len(re.findall(r"[lcrp]", spec.split("@{}")[-1] if "@{}" in spec else spec))
    counts = Counter()
    for line in body.split(r"\\"):
        line = line.strip()
        if not line or line.startswith(("\\toprule", "\\midrule", "\\botrule", "\\cmidrule",
                                       "\\multicolumn", "\\bottomrule")):
            continue
        counts[line.count("&") + 1] += 1
    print(f"tabular cols(approx {ncol}): row-width histogram {dict(counts)}")

# 7) 关键模板元素
for k in [r"\documentclass", r"\title[", r"\author*", r"\affil*", r"\abstract{", r"\keywords{",
          r"\maketitle", r"\backmatter", r"\bmhead", r"\begin{appendices}", r"\bibliography{refs}"]:
    print(f"{k:24s} {len(re.findall(re.escape(k), tex))}")

# 7b) adjustbox / resizebox 绝不能出现在单栏 table 中
#     sn-jnl.cls 把单栏 table 重定义为 tableorg -> center -> threeparttable，
#     而 threeparttable 会拦截 tabular 以测量表宽；adjustbox 采集表体时与之冲突，
#     编译会报 "\begin{threeparttable} ... ended by \end{tabular}" 并级联崩溃。
#     放不下单栏的表应当改用 table*（期刊模板明确推荐的做法）。
danger = []
for m in re.finditer(r"\\begin\{table\}(?:\[[^\]]*\])?(.*?)\\end\{table\}", tex, re.S):
    if "adjustbox" in m.group(1) or "resizebox" in m.group(1):
        danger.append(tex[: m.start()].count("\n") + 1)
print("adjustbox/resizebox inside single-column table:", danger or "none (OK)")

# 7c) sn-basic.bst 会把 @article 的 journal 字段逐字符过一遍 remove.dots，
#     静默删掉其中所有字面 "."（arXiv:2006.07159 -> arXiv:200607159）。
#     花括号保护无效——那个函数不跟踪分组。预印本必须用 @misc + howpublished。
#     另外 .bib 里 % 不是注释符，注释中若出现 @ 会被当成条目开头。
dotjournals = []
for m in re.finditer(r"@article\{([^,]+),(.*?)\n\}", bib, re.S):
    jm = re.search(r"journal\s*=\s*\{(.*?)\}", m.group(2), re.S)
    if jm and "." in jm.group(1):
        dotjournals.append((m.group(1), jm.group(1)))
print("article journal fields containing '.':", dotjournals or "none (OK)")
stray = [m.group(0)[:40] for m in re.finditer(r"%[^\n]*@\w+", bib)]
print("bib comments containing an at-sign:", stray or "none (OK)")

# 7d) 作者块：sn-jnl 用 \author*[n]{...} 标记通讯作者，应当恰好一位；
#     同时检查全文与投稿信里没有残留"单作者"措辞。
n_reg = len(re.findall(r"\\author(?!\*)\s*\[", tex))
n_star = len(re.findall(r"\\author\*\s*\[", tex))
print(f"authors: {n_reg} regular + {n_star} corresponding ->",
      "OK" if n_star == 1 else "CHECK (expected exactly one \\author*)")
for m in re.finditer(r"\\author(\*?)\s*\[[^\]]*\]\{((?:[^{}]|\{[^{}]*\})*)\}\s*\\email\{([^}]*)\}", tex):
    print(f"   {'*' if m.group(1) else ' '} {m.group(2).strip()} <{m.group(3)}>")
leftover = [ln.strip()[:60] for ln in tex.splitlines()
            if re.search(r"sole author|the author thanks|the author declares", ln, re.I)]
print("single-author phrasing left in main.tex:", leftover or "none (OK)")

letter = PAPER / "cover_letter.md"
if letter.exists():
    lt = letter.read_text(encoding="utf-8")
    bad = [ln.strip()[:60] for ln in lt.splitlines()
           if re.search(r"sole author|\bI am pleased\b|\bI study\b|\bI believe\b|\bI look forward\b", ln)]
    print("single-author phrasing left in cover_letter.md:", bad or "none (OK)")



n_tab = len(re.findall(r"\\begin\{table\}(?:\[[^\]]*\])?", tex))
n_tabstar = len(re.findall(r"\\begin\{table\*\}", tex))
n_adj = len(re.findall(r"\\begin\{adjustbox\}", tex))
print(f"tables: {n_tab} single-column | {n_tabstar} table* | {n_adj} adjustbox wrappers")
print("table/table* balance:", n_tab, n_tabstar,
      "| end*:", len(re.findall(r"\\end\{table\*\}", tex)), len(re.findall(r"\\end\{table\}", tex)))


# 8) label / ref 一致性
labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
refs = set()
for m in re.finditer(r"\\(?:ref|eqref)\{([^}]*)\}", tex):
    refs.update(k.strip() for k in m.group(1).split(",") if k.strip())
print("labels:", len(labels), "| refs:", len(refs))
print("DANGLING refs (no label):", sorted(refs - labels) or "none")
print("unreferenced labels:", sorted(labels - refs) or "none")
