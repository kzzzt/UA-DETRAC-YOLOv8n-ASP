# Cover Letter（投稿信）

> 用法：投稿系统里通常有一个 "Cover Letter" 文本框，把下面正文粘贴进去即可（方括号处替换）。
> 建议同时准备一份 PDF 版（可用 Word 排版后导出）。

---

**To the Editor-in-Chief**
*Applied Intelligence*

**Re: Submission of original research manuscript**

Dear Editor,

We are pleased to submit our manuscript entitled **"Parameter-Free Attention and a High-Resolution Detection Head for Small-Object Vehicle Detection: A Multi-Seed and Cross-Initialization Study on UA-DETRAC"** for consideration for publication in *Applied Intelligence*.

**What the paper does.** Small-object detection dominates the error budget in traffic-surveillance imagery, and the literature contains many "lightweight improvement" variants of YOLO detectors that combine an attention module, an extra high-resolution prediction head and a refined regression loss. We study three such modifications — SimAM (a parameter-free, energy-based attention module), an additional P2 prediction level, and the EIoU regression loss — on the UA-DETRAC benchmark, and we evaluate all six ablation variants under a protocol that is unusually strict for this literature: three random seeds, two initialization regimes (from scratch and from COCO-pretrained weights), two checkpoint-selection conventions (`best` and `last`), and two metric conventions (native Ultralytics and official COCO scoring), with Welch *t*-tests on every comparison.

**Why it is significant.** The central finding is quantitative rather than anecdotal, and it is actionable for practitioners. First, no modification improves overall mAP significantly in any of the four evaluation settings (largest gain `+0.0114` mAP@[0.5:0.95]); the seed-to-seed standard deviation of a nano-scale detector on this dataset is comparable to the effect size of the modifications, so single-run comparisons largely measure the seed. Second, two confounds that are widespread but rarely reported are quantified: reporting the `best` checkpoint instead of the final one changes the identity of the winning variant by up to `2.63` points of mAP@0.5, and COCO-pretrained initialization raises the baseline by `+0.0062` mAP@0.5 while shrinking the advantage of the modifications (P2: `+0.0113` → `+0.0017`). Third, the same checkpoint differs by `0.9` mAP@0.5 between the two standard scoring routines, a trap that silently corrupts tables. Against this background the paper does identify what survives scrutiny: the additional P2 prediction level is the most robust of the three modifications, leading a main column in three of four settings and delivering a statistically significant recall gain of `+0.0403` under pretrained initialization, at a documented cost of +51% GFLOPs.

**Why *Applied Intelligence*.** The journal's readership applies detection models under practical constraints, and the practical question addressed here — whether a lightweight modification is worth its cost, and how to tell — is directly relevant to that audience. The paper also contributes reusable methodology: an open-source pipeline that runs the multi-seed matrix, evaluates under both metric conventions without mixing them, performs the significance tests, and regenerates every table and figure from committed evaluation artefacts. We believe this combination of an applied benchmark study and a reproducibility toolchain fits the journal's scope.

**Reproducibility.** The complete source code, model configurations, evaluation artefacts, significance reports and the scripts that regenerate every table and figure are openly available at `https://github.com/kzzzt/UA-DETRAC-YOLOv8n-ASP`. The evaluation artefacts are committed to the repository, so all reported numbers can be reproduced in seconds without retraining.

**Declarations.** This manuscript is original, has not been published previously, and is not under consideration elsewhere. The authors declare no competing interests. No funding was received. The study uses only a publicly available, fully anonymized vehicle-detection dataset and involves no human or animal subjects. Z. Kang designed the study, implemented the model variants and the analysis pipeline, conducted all experiments and analyses, and wrote the manuscript; X. Liu supervised the study and revised the manuscript.

We thank you for considering this manuscript and look forward to your response.

Sincerely,

**Xiaoxiang Liu** (corresponding author)
School of Intelligent Systems Science and Engineering
Jinan University, Zhuhai Campus
206 Qianshan Road, Xiangzhou District, Zhuhai 519070, Guangdong, China
Email: tlxx@jnu.edu.cn

**Zhitong Kang** (first author)
Email: kangzt@stu2024.jnu.edu.cn

---

## 投稿前建议另外准备的（系统里会问）

| 项 | 建议 |
|----|------|
| Suggested Reviewers | 系统通常要求 3–5 位。建议找**做过 YOLO 改进但方法严谨**、或**写过检测复现性/统计显著性工作**的学者；避免选自己单位或有合作关系的 |
| Opposed Reviewers | 可选，一般不填 |
| Article type | Original Research（不要选 Review / Brief Report）|
| ORCID | 若已有，先在 orcid.org 注册并在系统里关联（Applied Intelligence 支持）|
| Highlights / Graphical abstract | Applied Intelligence 不强制；若系统允许，可用 `docs/figures/fig1_overall.png` 作图形摘要 |
| 推荐关键词 | 已定 6 个：object detection, small-object detection, attention mechanism, YOLOv8, UA-DETRAC, statistical significance |

## 关于投稿策略的实话

这篇稿子的定位是**「严谨评估 + 可复现工具链」**，不是「刷点」。所以投稿信里我刻意把
「负结果 + 量化出的三个普遍陷阱」当成卖点来写——这是它相对同题材论文真正的差异点。
如果编辑或审稿人期待的是 SOTA 提升，会有被拒风险；建议做好两件事：

1. **保留从零训练与预训练两套结果**（现在都有），任何一方被质疑都有另一套支撑；
2. 若被拒，主要修改方向是**补充跨模型验证**（把同一协议套到 YOLOv8s / YOLO11n 上），
   把结论从"某一组模块无效"提升为"这类轻量改进的效应量普遍小于种子噪声"，
   那时论文的说服力会明显上一个台阶。数据管线已经现成，主要是算力成本。
