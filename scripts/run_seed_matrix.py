"""按「变体 × 种子」矩阵批量训练，用于多种子重复实验（统计显著性 / 误差棒）。

变体定义与主实验完全一致（config / batch / 是否启用 EIOU）：

    baseline  : configs/yolov8n.yaml        batch 32
    simam     : configs/yolov8n-simam.yaml  batch 32
    p2        : configs/yolov8n-p2.yaml     batch 16
    eiou      : configs/yolov8n.yaml        batch 32  --eiou
    simam_p2  : configs/yolov8n-asp.yaml    batch 16
    asp       : configs/yolov8n-asp.yaml    batch 16  --eiou

运行名规则：``<prefix><variant>_s<seed>``（主实验的 seed=0 即不带 ``_s<seed>`` 后缀）。

用法:
    # 为 4 个核心变体补 2 个种子（共 8 次训练，约 2.5~3 天）
    python scripts/run_seed_matrix.py --variants baseline simam p2 simam_p2 --seeds 1 2

    # 全部 6 个变体 × 种子 1,2（共 12 次训练）
    python scripts/run_seed_matrix.py --seeds 1 2

    # 服务器上跑「COCO 预训练初始化」的整套矩阵（种子 0,1,2，前缀 pre_）
    python scripts/run_seed_matrix.py --seeds 0 1 2 --weights yolov8n.pt --prefix pre_

特性:
    - 已存在 ``weights/best.pt`` 的任务自动跳过，可随时中断/续跑；
    - 每次训练的完整日志写入 ``runs/seed_logs/<name>.log``；
    - 串行执行（单卡不会互相抢显存）；
    - ``--weights`` 指定的预训练初始化会写入每个运行目录下的 ``init.txt``，
      便于事后核对"某次运行是从头训练还是预训练初始化"。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VARIANTS: dict[str, dict] = {
    "baseline": dict(config="configs/yolov8n.yaml", batch=32, eiou=False),
    "simam": dict(config="configs/yolov8n-simam.yaml", batch=32, eiou=False),
    "p2": dict(config="configs/yolov8n-p2.yaml", batch=16, eiou=False),
    "eiou": dict(config="configs/yolov8n.yaml", batch=32, eiou=True),
    "simam_p2": dict(config="configs/yolov8n-asp.yaml", batch=16, eiou=False),
    "asp": dict(config="configs/yolov8n-asp.yaml", batch=16, eiou=True),
}


def run_state(name: str, runs_root: Path) -> str:
    """判断某次运行的状态：``done`` / ``partial`` / ``none``。

    判定依据：ultralytics 在训练**正常结束**时会剥离 checkpoint 里的优化器状态
    （日志中的 "Optimizer stripped from ..."）。因此：

        last.pt 仍含 optimizer  -> 训练被中断，可断点续训（partial）
        last.pt 已无 optimizer  -> 训练已正常结束（done）
        两者都不存在             -> 尚未开始（none）

    这样脚本重复执行时，已完成的会跳过、被中断的会自动续训。
    """
    wdir = runs_root / name / "weights"
    last, best = wdir / "last.pt", wdir / "best.pt"
    if not last.exists():
        return "done" if best.exists() else "none"
    try:
        import torch
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        return "partial" if ckpt.get("optimizer") is not None else "done"
    except Exception:
        return "done"  # 读不出来时按已完成处理，避免误覆盖已有结果


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="变体 × 种子 批量训练")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS),
                   choices=list(VARIANTS), help="要补哪些变体")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2],
                   help="要补的种子列表（0 为主实验，无需重复）")
    p.add_argument("--prefix", default="",
                   help="运行名前缀，如 pre_ 表示预训练初始化的一整套实验（默认空）")
    p.add_argument("--weights", default=None,
                   help="可选预训练权重（如 yolov8n.pt）；留空表示从零训练")
    p.add_argument("--force", action="store_true",
                   help="已完成的运行也强制从头重跑（默认跳过）")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--runs_root",
                   default=r"E:\资料\基座\学习兴趣组\runs\detect\runs\train",
                   help="训练输出根目录（用于判断任务是否已完成）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = PROJECT_ROOT / "runs" / "seed_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    runs_root = Path(args.runs_root)
    plan = [(v, s) for v in args.variants for s in args.seeds]

    # 先扫描状态，便于判断"这次要跑什么"
    states = {f"{args.prefix}{v}_s{s}": run_state(f"{args.prefix}{v}_s{s}", runs_root)
              for v, s in plan}
    label = {"done": "已完成（将跳过）", "partial": "训练中断（将断点续训）", "none": "未开始"}
    print(f"[计划] 共 {len(plan)} 次训练（前缀 {args.prefix!r}，"
          f"初始化 {'预训练 ' + str(args.weights) if args.weights else '从零训练'}）：")
    for v, s in plan:
        n = f"{args.prefix}{v}_s{s}"
        print(f"   {n:<16} {label[states[n]]}")

    for v, s in plan:
        name = f"{args.prefix}{v}_s{s}"
        state = states[name]

        if state == "done" and not args.force:
            print(f"[跳过] {name} 已完成")
            continue

        spec = VARIANTS[v]
        cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "train.py"),
            "--config", spec["config"],
            "--name", name,
            "--batch", str(spec["batch"]),
            "--seed", str(s),
            "--epochs", str(args.epochs),
            "--imgsz", str(args.imgsz),
        ]
        if spec["eiou"]:
            cmd.append("--eiou")
        if args.device:
            cmd += ["--device", args.device]

        resuming = state == "partial" and not args.force
        if resuming:
            # --resume 与 --weights 互斥：续训时权重来自 checkpoint
            cmd.append("--resume")
        elif args.weights:
            cmd += ["--weights", args.weights]

        # 记录初始化方式，便于事后核对（在训练前先写，避免中途失败丢信息）
        try:
            (runs_root / name).mkdir(parents=True, exist_ok=True)
            (runs_root / name / "init.txt").write_text(
                f"variant={v}\nseed={s}\nconfig={spec['config']}\n"
                f"batch={spec['batch']}\neiou={spec['eiou']}\n"
                f"weights={args.weights or 'scratch'}\n"
                f"epochs={args.epochs}\nimgsz={args.imgsz}\n",
                encoding="utf-8")
        except OSError as e:  # 只读或跨盘权限问题时不影响训练
            print(f"[提示] 无法写入 init.txt: {e}")

        log = log_dir / f"{name}.log"
        action = "续训" if resuming else "训练"
        print(f"\n[{action}] {name}  (config={spec['config']}, batch={spec['batch']}, "
              f"eiou={spec['eiou']}, weights={args.weights or 'scratch'})")
        print(f"       日志 -> {log}")
        with log.open("a" if resuming else "w", encoding="utf-8") as f:
            f.write(f"\n===== {'RESUME' if resuming else 'START'} {name} =====\n")
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT)).returncode
        print(f"[{'完成' if rc == 0 else '失败 rc=%d' % rc}] {name}")

    print("\n[全部结束] 下一步：对新增运行跑分场景评估，再运行 scripts/aggregate_results.py 汇总。")


if __name__ == "__main__":
    main()
