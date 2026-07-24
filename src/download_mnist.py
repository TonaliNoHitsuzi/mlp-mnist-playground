"""MNIST 数据集多镜像下载器。

torchvision 默认的 MNIST 下载源（LeCun 官网 / AWS S3）在国内经常超时失败。
本模块提供多镜像回退 + 文件大小校验，把 4 个 .gz 文件放到 torchvision 期望的位置，
之后 torchvision.datasets.MNIST(download=False) 就能直接读取，不再触发下载。

文件放置目录：<root>/MNIST/raw/
"""
from __future__ import annotations

import gzip
import os
import shutil
import sys
from pathlib import Path

# Windows 控制台默认 GBK 编码，遇到 ✓/✗ 等符号会崩溃，强制 UTF-8 输出
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests

# torchvision 加载 MNIST 需要的 4 个文件名 + 官方字节数（用于校验下载完整）
RESOURCES: list[tuple[str, int]] = [
    ("train-images-idx3-ubyte.gz", 9912422),
    ("train-labels-idx1-ubyte.gz", 28881),
    ("t10k-images-idx3-ubyte.gz", 1648877),
    ("t10k-labels-idx1-ubyte.gz", 4542),
]

# 多个镜像源，按优先级尝试。国内访问 googleapis 通常失败，S3 偶尔可用，
# HuggingFace 镜像（hf-mirror）是国内最稳定的兜底方案。
MIRRORS: list[str] = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://hf-mirror.com/datasets/ylecun/mnist/resolve/main/mnist/",
]

# 连接/读取超时（秒），避免某个卡住的源拖死整个下载
_CONNECT_TIMEOUT = 15
_READ_TIMEOUT = 60


def _raw_dir(root: str | Path) -> Path:
    """返回 torchvision 约定的 raw 子目录路径。"""
    return Path(root) / "MNIST" / "raw"


def _is_complete(root: str | Path) -> bool:
    """检查 4 个 .gz 文件是否都已存在且大小正确。"""
    d = _raw_dir(root)
    for name, size in RESOURCES:
        f = d / name
        if not f.exists() or f.stat().st_size != size:
            return False
    return True


def _download_one(url: str, dest: Path) -> bool:
    """从单个 URL 流式下载到 dest，带进度显示，成功返回 True。"""
    try:
        # 用 requests Session 复用连接，带浏览器 UA 避免被部分源拒绝
        with requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            stream=True,
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
        ) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded * 100 // total
                            # \r 回车回到行首，实现单行进度刷新
                            print(
                                f"\r    下载中 {pct:3d}% "
                                f"({downloaded // 1024}/{total // 1024} KB)",
                                end="",
                                flush=True,
                            )
            print(flush=True)  # 换行结束进度条
        return True
    except Exception as e:  # noqa: BLE001
        print(f"\n    [失败] {url} -> {e}", flush=True)
        return False


def ensure_mnist_downloaded(root: str | Path = "data") -> bool:
    """确保 MNIST 已下载完整。多镜像回退，成功返回 True。

    会顺带解压 .gz 文件（torchvision 部分版本需要解压后的 ubyte 文件）。
    """
    root = Path(root)
    raw = _raw_dir(root)
    raw.mkdir(parents=True, exist_ok=True)

    if _is_complete(root):
        print(f"[MNIST] 数据已存在于 {raw}，跳过下载。", flush=True)
        _extract_all(raw)
        return True

    print(f"[MNIST] 开始下载到 {raw} ...", flush=True)
    for name, expected_size in RESOURCES:
        dest = raw / name
        if dest.exists() and dest.stat().st_size == expected_size:
            print(f"  [跳过] {name} 已存在且大小正确", flush=True)
            continue
        # 单个文件多镜像尝试
        done = False
        for mirror in MIRRORS:
            url = mirror + name
            print(f"  尝试: {url}", flush=True)
            if _download_one(url, dest):
                if dest.stat().st_size == expected_size:
                    print(
                        f"    [成功] 大小校验通过 ({expected_size} bytes)",
                        flush=True,
                    )
                    done = True
                    break
                else:
                    print(
                        f"    [警告] 大小不符: "
                        f"得到 {dest.stat().st_size} / 期望 {expected_size}，换源重试",
                        flush=True,
                    )
                    dest.unlink(missing_ok=True)
        if not done:
            print(f"[MNIST] ✗ 文件 {name} 所有镜像均失败，请检查网络。", flush=True)
            print(
                "        手动方案：从 https://yann.lecun.com/exdb/mnist/ "
                f"下载这 4 个文件放到 {raw}",
                flush=True,
            )
            return False

    _extract_all(raw)
    print("[MNIST] ✓ 全部下载完成。", flush=True)
    return True


def _extract_all(raw: Path) -> None:
    """解压所有 .gz 文件（部分 torchvision 版本会读取解压后的 ubyte 文件）。"""
    for name, _ in RESOURCES:
        gz_path = raw / name
        out_path = raw / name[:-3]  # 去掉 .gz 后缀
        if gz_path.exists() and not out_path.exists():
            try:
                with gzip.open(gz_path, "rb") as fin, open(out_path, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
            except Exception:  # noqa: BLE001
                pass  # 解压失败不致命，torchvision 多数版本能直接读 .gz


if __name__ == "__main__":
    # 命令行直接运行此脚本即可预下载数据
    ok = ensure_mnist_downloaded("data")
    print("下载成功" if ok else "下载失败，请检查网络后重试")
