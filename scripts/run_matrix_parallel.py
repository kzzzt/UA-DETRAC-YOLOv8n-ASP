"""并行跑「变体 × 种子」矩阵：单卡上同时开 N 个训练进程，提高 GPU 利用率。

与 ``run_seed_matrix.py`` 的唯一区别是并发：串行版适合显存紧张的单卡，
本脚本适合大显存卡（如 4090 24GB）——YOLOv8n 级别的模型单进程吃不满卡，
开 2~3 个进程可显著缩短总时长。

调度策略:
    - 变体按显存需求分两类：``light``（baseline / simam / eiou，无 P2 头）
      与 ``heavy``（p2 / simam_p2 / asp，多一个 P2 检测层）。
    - 采用任务池而非批次屏障：谁先结束就先补新任务，避免互相等待。
    - 同时运行的 heavy 数量上限为 ``jobs - 1``（至少留一个槽给 light），
      防止两个大模型把显存挤爆。
    - 主循环结束后对失败任务做 ``--retries`` 轮串行重试（OOM 后单进程重跑通常能过）。

用法:
    # 单卡开 2 个进程，跑「从零训练」的种子 3,4,5
    python scripts/run_matrix_parallel.py --jobs 2 --seeds 3 4 5 \\
        --runs_root /root/autodl-tmp/runs/detect/runs/train

    # 单卡开 2 个进程，跑「COCO 预训练初始化」矩阵
    python scripts/run_matrix_parallel.py --jobs 2 --seeds 0 1 2 \\
        --prefix pre_ --weights /root/autodl-tmp/weights/yolov8n.pt \\
        --runs_root /root/autodl-tmp/runs/detect/runs/train
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_seed_matrix import VARIANTS, run_state  # noqa: E402

# 显存需求分组（P2 变体多一个高分辨率检测层，显存约为 light 的 1.5~2 倍）
HEAVY = {"p2", "simam_p2", "asp"}


def build_cmd(variant: str, seed: int, args, resume: bool) -> list[str]:
    spec = VARIANTS[variant]
    # --batch 可强制覆盖所有变体的 batch（服务器实测：瓶颈是"每轮迭代开销"而非每张图，
    # 大显存卡上把 P2 系列从 batch16 提到 32 可让单轮耗时接近减半，且全表 batch 更统一）
    batch = getattr(args, "batch", None) or spec["batch"]
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "train.py"),
        "--config", spec["config"],
        "--name", f"{args.prefix}{variant}_s{seed}",
        "--batch", str(batch),
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--imgsz", str(args.imgsz),
    ]
    if spec["eiou"]:
        cmd.append("--eiou")
    if args.data:
        cmd += ["--data", args.data]
    if args.device:
        cmd += ["--device", args.device]
    if getattr(args, "workers", None):
        cmd += ["--workers", str(args.workers)]
    if resume:
        cmd.append("--resume")  # 与 --weights 互斥：续训权重取自 checkpoint
    elif args.weights:
        cmd += ["--weights", args.weights]
    return cmd


def build_queue(variants: list[str], seeds: list[int]) -> list[tuple[str, int]]:
    """按「每个种子内 light/heavy 交替」的顺序排队，便于任务池混搭。"""
    light = [v for v in variants if v not in HEAVY]
    heavy = [v for v in variants if v in HEAVY]
    queue: list[tuple[str, int]] = []
    for s in seeds:
        for i in range(max(len(light), len(heavy))):
            if i < len(light):
                queue.append((light[i], s))
            if i < len(heavy):
                queue.append((heavy[i], s))
    return queue


def pick(queue: list[tuple[str, int]], running: list, jobs: int):
    """从队列里挑下一个任务：保证 heavy 并发数不超过 jobs-1。"""
    heavy_running = sum(1 for r in running if r["heavy"])
    heavy_limit = max(1, jobs - 1)
    for i, (variant, _seed) in enumerate(queue):
        if variant in HEAVY and heavy_running >= heavy_limit:
            continue
        return queue.pop(i)
    return queue.pop(0)


def write_init(runs_root: Path, name: str, variant: str, seed: int, args) -> None:
    spec = VARIANTS[variant]
    try:
        (runs_root / name).mkdir(parents=True, exist_ok=True)
        (runs_root / name / "init.txt").write_text(
            f"variant={variant}\nseed={seed}\nconfig={spec['config']}\n"
            f"batch={getattr(args, 'batch', None) or spec['batch']}\neiou={spec['eiou']}\n"
            f"weights={args.weights or 'scratch'}\n"
            f"epochs={args.epochs}\nimgsz={args.imgsz}\n"
            f"data={args.data or 'configs/ua_detrac.yaml'}\n"
            f"workers={getattr(args, 'workers', None) or 8}\n", encoding="utf-8")
    except OSError as e:
        print(f"[提示] 无法写入 init.txt: {e}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="并行跑变体 × 种子矩阵")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--prefix", default="")
    p.add_argument("--weights", default=None)
    p.add_argument("--jobs", type=int, default=2, help="并发训练进程数")
    p.add_argument("--retries", type=int, default=1, help="失败任务串行重试轮数")
    p.add_argument("--force", action="store_true")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--data", default=None,
                   help="数据集 YAML（默认用 train.py 的 configs/ua_detrac.yaml；"
                        "服务器上可用 configs/ua_detrac_monitor.yaml 以缩短每轮验证）")
    p.add_argument("--workers", type=int, default=None,
                   help="每个训练进程的 DataLoader 进程数（多核服务器建议 16~24）")
    p.add_argument("--batch", type=int, default=None,
                   help="强制覆盖所有变体的 batch size（默认用 run_seed_matrix.VARIANTS 里的值）")
    p.add_argument("--runs_root", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    log_dir = PROJECT_ROOT / "runs" / "seed_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    queue = build_queue(args.variants, args.seeds)
    print(f"[计划] {len(queue)} 次训练，并发 {args.jobs}，"
          f"初始化 {('预训练 ' + args.weights) if args.weights else '从零训练'}",
          flush=True)

    failed: list[tuple[str, int]] = []
    running: list[dict] = []
    started = finished = 0
    t_start = time.time()

    def start(variant: str, seed: int) -> None:
        nonlocal started
        name = f"{args.prefix}{variant}_s{seed}"
        st = "none" if args.force else run_state(name, runs_root)
        if st == "done":
            print(f"[跳过] {name} 已完成", flush=True)
            return
        resume = st == "partial"
        write_init(runs_root, name, variant, seed, args)
        cmd = build_cmd(variant, seed, args, resume)
        log = log_dir / f"{name}.log"
        fh = log.open("a" if resume else "w", encoding="utf-8")
        fh.write(f"\n===== {'RESUME' if resume else 'START'} {name} "
                 f"({time.strftime('%F %T')}) =====\n")
        fh.flush()
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                             cwd=str(PROJECT_ROOT))
        running.append(dict(p=p, name=name, variant=variant, seed=seed, fh=fh,
                            heavy=variant in HEAVY, t0=time.time()))
        started += 1
        print(f"[启动] {name:<18} pid={p.pid} 并发={len(running)} "
              f"({'续训' if resume else '训练'})", flush=True)

    while queue or running:
        while queue and len(running) < args.jobs:
            variant, seed = pick(queue, running, args.jobs)
            start(variant, seed)

        time.sleep(10)
        for r in list(running):
            rc = r["p"].poll()
            if rc is None:
                continue
            r["fh"].close()
            running.remove(r)
            el = (time.time() - r["t0"]) / 3600
            if rc == 0:
                finished += 1
                print(f"[完成] {r['name']:<18} 用时 {el:.2f} h  "
                      f"(累计 {finished}/{started})", flush=True)
            else:
                failed.append((r["variant"], r["seed"]))
                print(f"[失败 rc={rc}] {r['name']:<18} 用时 {el:.2f} h", flush=True)

    for attempt in range(1, args.retries + 1):
        if not failed:
            break
        print(f"\n[重试 {attempt}] 串行重跑 {len(failed)} 个失败任务", flush=True)
        retry, failed = failed, []
        for variant, seed in retry:
            name = f"{args.prefix}{variant}_s{seed}"
            st = run_state(name, runs_root)
            cmd = build_cmd(variant, seed, args, resume=(st == "partial"))
            log = log_dir / f"{name}.log"
            with log.open("a", encoding="utf-8") as fh:
                fh.write(f"\n===== RETRY {attempt} {name} ({time.strftime('%F %T')}) =====\n")
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    cwd=str(PROJECT_ROOT)).returncode
            print(f"[重试{'完成' if rc == 0 else f'失败 rc={rc}'}] {name}", flush=True)
            if rc == 0:
                finished += 1
            else:
                failed.append((variant, seed))

    print(f"\n[全部结束] 成功 {finished}，失败 {len(failed)}，"
          f"总用时 {(time.time() - t_start) / 3600:.2f} h", flush=True)
    if failed:
        print("  失败清单: " + ", ".join(f"{v}_s{s}" for v, s in failed), flush=True)


if __name__ == "__main__":
    main()
