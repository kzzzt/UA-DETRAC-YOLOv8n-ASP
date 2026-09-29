"""YOLOv8n-ASP 训练脚本。

基于 ultralytics 的 YOLO 接口，加载本项目定义的模型结构进行训练。

模型结构通过 YAML 配置表达（由 `models` 包注册 C2fSimAM 后即可解析）：
    - configs/yolov8n.yaml       基线（原始 YOLOv8n）
    - configs/yolov8n-simam.yaml 仅 Neck 加 SimAM（消融 A）
    - configs/yolov8n-p2.yaml    仅新增 P2 小目标层（消融 S/P）
    - configs/yolov8n-asp.yaml   组合：SimAM + P2（核心改进模型）

EIOU 损失不属于网络结构，通过 ``--eiou`` 开关在 loss 层打补丁启用。

用法:
    # 训练基线
    python scripts/train.py --config configs/yolov8n.yaml --name baseline

    # 训练核心改进模型（SimAM + P2 + EIOU）
    python scripts/train.py --config configs/yolov8n-asp.yaml \
        --name yolov8n-asp --eiou

    # 消融：仅 SimAM
    python scripts/train.py --config configs/yolov8n-simam.yaml \
        --name yolov8n-simam

    # 消融：仅 P2 小目标层
    python scripts/train.py --config configs/yolov8n-p2.yaml \
        --name yolov8n-p2

    # 加载预训练权重（可选）
    python scripts/train.py --config configs/yolov8n-asp.yaml \
        --name yolov8n-asp --weights yolov8n.pt
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 UA-DETRAC YOLO 模型")
    parser.add_argument("--config", default="configs/yolov8n-asp.yaml",
                        help="模型结构 YAML 路径")
    parser.add_argument("--data", default="configs/ua_detrac.yaml",
                        help="数据集 YAML 路径")
    parser.add_argument("--name", default="yolov8n-asp",
                        help="实验名称（用于输出目录）")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=16,
                        help="批大小")
    parser.add_argument("--device", default=None,
                        help="设备：0=GPU / cpu=CPU，留空自动选择（无 GPU 自动用 CPU）")
    parser.add_argument("--patience", type=int, default=20,
                        help="早停耐心值")
    parser.add_argument("--workers", type=int, default=None,
                        help="DataLoader 进程数（留空用 ultralytics 默认 8；"
                             "多核服务器可调大以缓解数据加载瓶颈）")
    parser.add_argument("--cache", action="store_true",
                        help="把解码后的图像缓存到内存（'ram'），可显著加速，"
                             "但需要约 120 GB 内存，仅在内存充足的机器上使用")
    parser.add_argument("--weights", default=None,
                        help="可选预训练权重路径（如 yolov8n.pt）")
    parser.add_argument("--eiou", action="store_true",
                        help="启用 EIOU 边界框回归损失（替换 CIoU）")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子（多种子重复实验用，需配合不同 --name）")
    parser.add_argument("--resume", nargs="?", const=True, default=False,
                        help="断点续训：--resume 从该 name 的 last.pt 续训；"
                             "--resume <权重路径> 从指定权重续训（勿与 --weights 同用）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    data_path = Path(args.data)
    if not config_path.exists():
        raise FileNotFoundError(f"模型配置文件不存在: {config_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"数据配置文件不存在: {data_path}")

    # 延迟导入，确保 models 包已注册自定义模块（C2fSimAM 等）
    # noinspection PyUnresolvedReferences
    import models  # 触发 models/__init__.py 注册
    from ultralytics import YOLO

    # 清理已加载的缓存（防止重复实验间模块缓存冲突）
    import ultralytics
    if hasattr(ultralytics.nn.tasks, "clear_model_cache"):
        ultralytics.nn.tasks.clear_model_cache()

    # 可选：启用 EIOU 损失
    if args.eiou:
        from models.eiou_loss import patch_ultralytics_eiou
        ok = patch_ultralytics_eiou()
        if ok:
            print("[损失] 已启用 EIOU 边界框回归损失（替换 CIoU）。")
        else:
            print("[警告] EIOU 补丁未生效，回退为默认 CIoU。")

    print(f"[模型结构] {config_path}")
    print(f"[数据配置] {data_path}")
    print(f"[实验名称] {args.name}")

    model = YOLO(str(config_path))

    # 可选：加载预训练权重（P2 扩展层、检测头等形状不匹配的部分由 ultralytics 自动跳过）
    if args.weights and not args.resume:
        weights_path = Path(args.weights)
        if weights_path.exists():
            print(f"[加载] 预训练权重: {weights_path}")
            model.load(str(weights_path))
        elif weights_path.suffix in (".pt", ".pth") and len(weights_path.parts) == 1:
            # 只给了文件名（如 yolov8n.pt）→ 交给 ultralytics 自动下载
            print(f"[加载] 预训练权重（由 ultralytics 自动下载）: {args.weights}")
            model.load(args.weights)
        else:
            raise FileNotFoundError(f"预训练权重不存在: {weights_path}")

    # 裸 --resume（布尔 True）时，ultralytics 只支持从「自身已加载的 ckpt」续训，
    # 而本脚本用 YAML 构建模型（无 ckpt），需先把它解析成 last.pt 的路径字符串。
    if args.resume is True:
        from types import SimpleNamespace
        from ultralytics.cfg import get_save_dir
        sd = get_save_dir(SimpleNamespace(
            project="runs/train", name=args.name, task="detect",
            mode="train", exist_ok=True, save_dir=None,
        ))
        last = sd / "weights" / "last.pt"
        if last.exists():
            args.resume = str(last)
            print(f"[续训] 从 {last} 恢复训练")
        else:
            print(f"[警告] 未找到 {last}，无法续训，将从头训练")
            args.resume = False

    train_kwargs = dict(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        name=args.name,
        project="runs/train",
        exist_ok=True,
        seed=args.seed,
        resume=args.resume,
    )
    if args.workers is not None:
        train_kwargs["workers"] = args.workers
    if args.cache:
        train_kwargs["cache"] = "ram"

    model.train(**train_kwargs)

    print(f"\n[完成] 训练结束，结果保存在 {model.trainer.save_dir}")


if __name__ == "__main__":
    main()