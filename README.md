# YOLOv8n-ASP: SimAM + P2 + EIoU for Small-Object Vehicle Detection on UA-DETRAC

Code, configurations and evaluation artefacts for the paper

> **Parameter-Free Attention and a High-Resolution Detection Head for Small-Object Vehicle Detection: A Multi-Seed and Cross-Initialization Study on UA-DETRAC**

submitted to *Applied Intelligence*.

---

## What this repository contains

Six detector variants built on YOLOv8n, trained and evaluated under a fully controlled,
multi-seed protocol:

| Variant | Composition | Params | GFLOPs |
|---------|-------------|--------|--------|
| `baseline` | YOLOv8n | 3,011,433 | 8.2 |
| `simam` | + SimAM after each head feature map (P3–P5) | 3,011,433 | 8.2 |
| `p2` | + P2 prediction level | 2,926,956 | 12.4 |
| `eiou` | + EIoU regression loss | 3,011,433 | 8.2 |
| `simam_p2` | + SimAM + P2 (SimAM on P2–P5) | 2,926,956 | 12.4 |
| `asp` | + SimAM + P2 + EIoU | 2,926,956 | 12.4 |

Key property of the attention module: **SimAM is implemented as a standalone,
channel-preserving layer placed after each detection-head feature map and before the
detection head.** It therefore adds *zero* parameters and can be parsed by the generic
module-construction path — no modification of the model-parsing whitelist is required.

## Headline results

* Across four evaluation settings (two initializations × two checkpoint conventions),
  **no variant improves overall mAP significantly** (largest gain `+0.0114`
  mAP@[0.5:0.95]; Welch *t*-test *p* > 0.05).
* The **P2 prediction level is the most robust** modification: it leads a main column in
  three of four settings and yields a significant recall gain (`+0.0403`, *p* < 0.05) under
  pretrained initialization, at the cost of +51% GFLOPs.
* **EIoU is never beneficial** on this dataset and is significantly negative in two
  scene-level comparisons.
* **COCO-pretrained initialization raises the baseline by `+0.0062` mAP@0.5** and shrinks
  the advantage of the modifications (P2: `+0.0113` → `+0.0017`).
* The same checkpoint can differ by **0.9 mAP@0.5** between the native Ultralytics scoring
  protocol and the official COCO protocol; tables must not mix the two.

All numbers are means ± standard deviation over **three random seeds (0, 1, 2)**.

## Repository layout

```
configs/     model YAMLs (baseline / simam / p2 / asp), dataset YAMLs
models/      SimAM layer, EIoU loss + training patch, reference C2f-SimAM implementation
scripts/     training, evaluation, significance testing, table & figure generation
utils/       dataset path resolution, weather-scene annotations
docs/        engineering log, paper tables, conclusions, figures
results/     evaluation CSVs and significance reports (the numbers used in the paper)
paper/       LaTeX source of the manuscript (Springer Nature template)
```

## Installation

```bash
pip install -r requirements.txt          # ultralytics, torch, scipy, faster-coco-eval, ...
```

Evaluation with size-bucketed AP additionally requires `faster-coco-eval`:

```bash
pip install "faster-coco-eval>=1.6.7"
```

## Data preparation

1. Download UA-DETRAC from the official benchmark site.
2. Convert the XML annotations to YOLO format (sequence-disjoint split):

```bash
python scripts/convert_detrac.py --detrac_dir /path/to/DETRAC_raw --val_sequences val_list.txt
python scripts/classify_weather.py          # reads the official sence_weather field
```

3. Point Ultralytics at your dataset root (`datasets_dir` in `~/.config/Ultralytics/settings.json`),
   or edit `path:` in `configs/ua_detrac.yaml`.

## Training

```bash
# one variant
python scripts/train.py --config configs/yolov8n-asp.yaml --name simam_p2 --batch 16 --seed 0

# the full variant x seed matrix (serial, resumable)
python scripts/run_seed_matrix.py --seeds 0 1 2 --runs_root /path/to/runs/train

# or the same matrix with concurrent jobs on a large-memory GPU
python scripts/run_matrix_parallel.py --seeds 0 1 2 --jobs 3 --workers 7 \
    --runs_root /path/to/runs/train
```

The nominal batch size is fixed at `nbs=64`, so batch 16 and batch 32 configurations have
**identical effective batch size, optimizer-step count and learning-rate schedule**.

## Evaluation

```bash
# overall + per-weather-scene (native Ultralytics protocol)
python scripts/eval_scenes.py  --models baseline simam p2 eiou simam_p2 asp \
    --weights-file last.pt --runs_root /path/to/runs/train --out_csv results/summary.csv

# per-object-size AP (official COCO protocol)
python scripts/eval_by_size.py --models baseline simam p2 eiou simam_p2 asp \
    --weights-file last.pt --runs_root /path/to/runs/train --out_csv results/ap_by_size.csv

# Welch t-tests against the baseline
python scripts/analyze_significance.py --scene_csv results/summary.csv

# regenerate every table / figure in the paper
python scripts/make_paper_tables.py  --out docs/论文表格_自动生成.md
python scripts/make_paper_figures.py --out_dir docs/figures
```

## Reproducing the paper's numbers without retraining

The complete evaluation artefacts are committed under `results/`, so every table and figure
can be regenerated in seconds:

```bash
python scripts/make_paper_tables.py \
    --scratch_scene results/summary_all.csv \
    --scratch_size  results/ap_by_size_all.csv \
    --pre_scene     results/summary_pre_best.csv \
    --pre_size      results/ap_by_size_pre_best.csv \
    --out           docs/论文表格_自动生成.md
python scripts/make_paper_figures.py --out_dir docs/figures
```

## Notes on paths

Several scripts keep the authors' local output directory as the default value of
`--runs_root` (a Windows path). Pass `--runs_root` explicitly, as shown above, or use
`scripts/run_seed_matrix.py --runs_root <your dir>` when training.

## Citation

If you use this code, please cite the paper (reference to be completed upon publication)
and the UA-DETRAC benchmark:

```bibtex
@article{wen2020uadetrac,
  author  = {Wen, Longyin and Du, Dawei and Cai, Zhaowei and Lei, Zhen and Chang, Ming-Ching
             and Qi, Honggang and Lim, Jongwoo and Yang, Ming-Hsuan and Lyu, Siwei},
  title   = {{UA-DETRAC}: A New Benchmark and Protocol for Multi-Object Detection and Tracking},
  journal = {Computer Vision and Image Understanding},
  volume  = {193},
  pages   = {102907},
  year    = {2020}
}
```

## License

MIT (see `LICENSE`). The LaTeX class file `sn-jnl.cls` and the bibliography styles under
`paper/` are distributed by Springer Nature and are **not** covered by this license.
