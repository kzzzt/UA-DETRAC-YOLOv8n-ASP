"""UA-DETRAC 数据下载与解压脚本。

UA-DETRAC 官方数据需从官网申请后下载，典型压缩包包括:
    - DETRAC-Train-Annotations-XML.zip
    - DETRAC-Test-Annotations-XML.zip
    - DETRAC-train-data.zip
    - DETRAC-test-data.zip

本脚本支持:
    1. 从 URL 列表批量下载（若用户已获直链）。
    2. 解压到指定目录。
    3. 校验下载产物是否完整。

用法:
    # 默认：从本地已放置的压缩包解压
    python scripts/download_data.py --data_dir ./data

    # 从 URL 下载
    python scripts/download_data.py --data_dir ./data --urls urls.txt

说明:
    UA-DETRAC 数据集版权归原作者所有，请通过官方渠道获取。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# 常见官方压缩包名
EXPECTED_ARCHIVES = [
    "DETRAC-Train-Annotations-XML.zip",
    "DETRAC-Test-Annotations-XML.zip",
    "DETRAC-train-data.zip",
    "DETRAC-test-data.zip",
]

# 官方解压后的目录（UA-DETRAC 标准结构）
EXPECTED_DIRS = [
    "DETRAC-Train-Annotations-XML",
    "DETRAC-Test-Annotations-XML",
    "Insight-MVT_Annotation_Train",
    "Insight-MVT_Annotation_Test",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载并解压 UA-DETRAC 数据集")
    parser.add_argument("--data_dir", default="./data", help="数据根目录")
    parser.add_argument("--urls", default=None,
                        help="每行一个 URL 的文本文件，用于批量下载")
    parser.add_argument("--keep_zip", action="store_true",
                        help="保留压缩包，默认解压后删除")
    parser.add_argument("--no_checksum", action="store_true",
                        help="跳过压缩包完整性校验")
    return parser.parse_args()


def download_file(url: str, dest: Path) -> None:
    """使用 curl 下载文件。"""
    print(f"[下载] {url} -> {dest}")
    cmd = ["curl", "-L", "-C", "-", "-o", str(dest), url]
    subprocess.run(cmd, check=True)


def extract_zip(archive: Path, dest_dir: Path) -> None:
    """解压 zip 压缩包。"""
    print(f"[解压] {archive} -> {dest_dir}")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest_dir)


def resolve_archives(data_dir: Path) -> list[Path]:
    """返回数据目录下已有的官方压缩包路径列表。"""
    archives: list[Path] = []
    for name in EXPECTED_ARCHIVES:
        p = data_dir / name
        if p.exists():
            archives.append(p)
    return archives


def extract_all(data_dir: Path, keep_zip: bool) -> None:
    """解压当前目录下的所有官方压缩包。"""
    archives = resolve_archives(data_dir)
    if not archives:
        print("[提示] 未在数据目录找到官方压缩包，"
              "请先手动下载并放入该目录，或使用 --urls 指定下载地址。")
        return

    for archive in archives:
        extract_zip(archive, data_dir)
        if not keep_zip:
            archive.unlink()
            print(f"[清理] 已删除压缩包 {archive}")


def sanity_check(data_dir: Path) -> dict[str, bool]:
    """校验数据集关键目录是否存在（官方 UA-DETRAC 解压结构）。

    官方解压后通常为:
        DETRAC-Train-Annotations-XML/  训练标注
        Insight-MVT_Annotation_Train/   训练图像
        Insight-MVT_Annotation_Test/    测试图像（无公开 GT，仅推理用）
        DETRAC-Test-Annotations-XML/    测试标注（官方通常不公开，缺失属正常）
    """
    checks = {
        "train_xml": (data_dir / "DETRAC-Train-Annotations-XML").is_dir(),
        "train_imgs": (data_dir / "Insight-MVT_Annotation_Train").is_dir()
                       or (data_dir / "DETRAC-train-data" / "Insight-MVT_Annotation_Train").is_dir(),
        "test_imgs": (data_dir / "Insight-MVT_Annotation_Test").is_dir()
                      or (data_dir / "DETRAC-test-data").is_dir(),
        "test_xml": (data_dir / "DETRAC-Test-Annotations-XML").is_dir(),
    }
    return checks


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. 若指定 URL 文件，逐一下载
    if args.urls:
        urls_path = Path(args.urls)
        if not urls_path.exists():
            print(f"[错误] URL 文件不存在: {urls_path}")
            sys.exit(1)
        urls = [line.strip() for line in urls_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")]
        for url in urls:
            name = url.rstrip("/").split("/")[-1]
            download_file(url, data_dir / name)

    # 2. 解压
    extract_all(data_dir, args.keep_zip)

    # 3. 校验
    if not args.no_checksum:
        checks = sanity_check(data_dir)
        print("\n[校验] 数据集完整性:")
        for k, ok in checks.items():
            print(f"  - {k}: {'OK' if ok else 'MISSING'}")
        # 训练所需 = train_xml + train_imgs；test_xml 官方通常不公开，缺失不影响训练
        if checks["train_xml"] and checks["train_imgs"]:
            print("[成功] 训练数据就绪，可进入格式转换步骤"
                  "（test_xml 缺失属正常，验证集用 train 切分）。")
        else:
            print("[警告] 训练数据缺失，请确认下载产物完整。")


if __name__ == "__main__":
    main()