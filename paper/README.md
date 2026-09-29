# 投稿稿件（Springer Nature 模板）

本目录是可直接编译的 LaTeX 初稿，使用期刊提供的 `sn-jnl.cls` 模板（v3.1, December 2024）。

```
paper/
├── main.tex                  ← 正文（单文件，模板要求不得 \input 拆分）
├── refs.bib                  ← 参考文献（37 条，全部为真实文献，已全部被引用）
├── sn-jnl.cls                ← 期刊模板类文件（原样保留）
├── sn-mathphys-num.bst       ← 编号制参考文献样式（与 documentclass 的 sn-mathphys-num 对应）
├── bst/                      ← 模板附带的其它样式（备用）
├── figures/                  ← 三张插图（由 scripts/make_paper_figures.py 生成）
└── supplementary/            ← 补充材料：全部逐次运行指标与显著性表
```

## 一、如何编译

本机没有安装 LaTeX，需用下面任一方式编译：

**方式 A：Overleaf（最省事）**
1. 上传 `paper_sn_submission.zip`（在仓库根目录，已用 Python 重打包，路径为正斜杠）；
2. Overleaf → New Project → Upload Project；
3. 编译器选 **pdfLaTeX**，主文件 `main.tex`，点击 Recompile。

**方式 B：本地 TeX Live / MiKTeX**
```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

### 已修复的编译问题（2026-09-29）

| 报错 / 警告 | 原因 | 修复 |
|------------|------|------|
| `File 'figures/fig1_overall.png' not found`（×3）| 旧 zip 由 PowerShell `Compress-Archive` 生成，条目名是**反斜杠**（`figures\fig1.png`），Overleaf 在 Linux 上解压后目录结构丢失 | 改用 `python paper/_package.py` 打包（正斜杠）；同时正文加 `\graphicspath{{figures/}{./}}` 并把 `\includegraphics` 改为不带扩展名的写法，图片放 `figures/` 或与 `main.tex` 同级都能找到 |
| `Font shape 'U/rsfs/m/n' not available`（×3）| 模板预置的 `mathrsfs` 宏包在本机字体集下缺少 Type1 字体，而正文并未使用 `\mathscr` | 删除 `\usepackage{mathrsfs}` |
| `Overfull \hbox` 表格超出版心（表 3、表 4、表 6 均出现过）| 7 列表 / 9 列表 / 6 列表的内容宽于版心 | **全部 7 张表**统一用 `\begin{adjustbox}{max width=...}` 包裹：`table*` 用 `\textwidth`、单栏 `table` 用 `\columnwidth`（**只在超宽时缩小，不会把小表放大**）；并统一加了 `\centering` |
| `hyperref Warning: Difference (N) between bookmark levels ...`（6 条）| `sn-jnl.cls` 把 `\bmhead` 定义为 `\@startsection` 的**第 5 级（subparagraph）**，从 1 级的 section 直接跳到 5 级 | 在 hyperref 之后加载 `\usepackage{bookmark}` 重建 PDF 书签树，警告消失且书签保留 |
| `Underfull \hbox (badness 1048) ... lines 42--42` | `\maketitle` 处的标题/作者块断行留白 | 纯外观，可忽略 |
| `Underfull \vbox (badness 10000) has occurred while \output is active` | 某页纵向拉伸（浮动体排布所致）| 纯外观，可忽略；若要消除可微调浮动体位置或增删文字 |

> 若你已经在 Overleaf 上建了旧项目：最简单的做法是删掉旧项目、重新上传新 zip；
> 或手动在 Overleaf 里新建 `figures` 文件夹，把本目录 `figures/` 下三张 PNG 传上去。

> 若 BibTeX 报找不到 `sn-mathphys-num.bst`，把 `bst/` 里的同名文件复制到 `main.tex` 同目录
> （本目录已预先复制了一份，正常情况无需处理）。

## 二、投稿信息现状（Applied Intelligence）

| 项 | 值 | 状态 |
|----|-----|------|
| 作者 | Zhitong Kang（康智童，第一作者）+ Xiaoxiang Liu（刘晓翔，通讯作者）| ✅ 已填 |
| 邮箱 | kangzt@stu2024.jnu.edu.cn（一作）；tlxx@jnu.edu.cn（通讯，`\author*` 标记）| ✅ 已填 |
| 单位 | School of Intelligent Systems Science and Engineering, Jinan University（暨南大学珠海校区智能科学与工程学院）| ✅ 已填 |
| 地址 | 206 Qianshan Road, Xiangzhou District, Zhuhai 519070, Guangdong, China | ⚠️ 请核对（前山路 206 号 / 519070）|
| 参考文献制式 | `sn-basic` + `Numbered`（Applied Intelligence 用编号制 Springer Basic）| ✅ 已切换 |
| 摘要 | 199 词（该刊上限通常 250 词）| ✅ |
| 关键词 | 6 个 | ✅ |
| Funding | Not applicable（无基金）| ✅ |
| Declarations 其余各项 | 按 SN 模板要求逐项给出 | ✅ |
| 投稿信 | `cover_letter.md` | ✅ 已起草 |
| 代码开源 | https://github.com/kzzzt/UA-DETRAC-YOLOv8n-ASP | ✅ 地址已写入正文与投稿信；仓库待推送（见 `PUSH_TO_GITHUB.md`）|

> **投稿前只剩一件事**：把仓库推送到 GitHub（`../UA-DETRAC-YOLOv8n-ASP/PUSH_TO_GITHUB.md`）。
> 其余信息均已填妥，无需再改。

## 三、稿件结构与核心论点

| 章节 | 内容 |
|------|------|
| 1 Introduction | 小目标检测背景；指出"单种子 + 用验证集选最优权重再报验证集"两个普遍问题；列出四点贡献 |
| 2 Related Work | 实时检测器与多尺度预测、注意力模块（SE/CBAM/ECA/CA/GCNet/SimAM）、IoU 损失族（GIoU/DIoU/CIoU/EIoU）、UA-DETRAC、检测论文中的统计规范 |
| 3 Proposed Method | SimAM 能量函数与闭式解（式 1–3）、独立层插入策略、P2 检测头、EIoU（式 4）、六个变体定义 |
| 4 Experimental Setup | 数据集与划分、两种训练机制（从零 100 轮 / 预训练 25 轮）、**两套评估口径的规则**、Welch t 检验 |
| 5 Results | 主表（表 3）、消融、分场景（表 4）、分尺寸（表 5）、收敛与选点、跨初始化稳健性（表 6） |
| 6 Discussion | 结论边界、为什么增益小、五条实践建议、五条局限 |
| 7 Conclusion | 主结论 + 方法学建议 |
| 附录 A/B | 可复现细节（含环境回退说明）、逐次运行结果 |

**核心论点（每句都有数据支撑）**

1. 四种评估设置下整体差异**全部不显著**（最大 +0.0114 mAP@[0.5:0.95]，Welch $p>0.05$）；
2. **P2 检测头最稳健**（四个设置里三个第一；预训练 last 下召回 +0.0403\* 显著），代价是 GFLOPs +51%；
3. **EIoU 始终无正增益**，两处场景级显著为负；
4. **SimAM 零参数**，但整体中性，是"小目标获益 / 中目标受损"的权衡；
5. 三项改进**不可叠加**（加 EIoU 后 ASP 反而是最弱）；
6. 预训练初始化把 baseline 抬高 +0.0062、把 P2 的增益从 +0.0113 压到 +0.0017；
7. 同一 checkpoint 在两套评估口径下差 0.9 个点，表格不可混用。

## 四、数据可追溯性

正文中每个数字都能追溯到本仓库的评估产物：

| 正文位置 | 数据来源 |
|----------|---------|
| 表 3、表 4（从零训练）| `runs/eval/summary_all.csv`、`runs/eval/significance_report.md` |
| 表 5（分尺寸）| `runs/eval/ap_by_size_all.csv`、`runs/eval/significance_by_size.md` |
| 表 6、附录 B（预训练）| `runs/eval_server/summary_pre_{best,last}.csv`、`ap_by_size_pre_{best,last}.csv` |
| 显著性标记 | `runs/eval_server/significance_pre_{scene,size}_{best,last}.md` |
| 图 1–3 | `scripts/make_paper_figures.py`（读上述 CSV 生成）|

这些 CSV 已复制到 `supplementary/`，可直接作为补充材料提交。
