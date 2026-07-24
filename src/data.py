"""MNIST 数据加载器。

职责：
1. 调用 download_mnist 确保数据本地可用（多镜像回退）
2. 应用标准化（用 MNIST 的经验均值/方差）
3. 从训练集切分出验证集（用于训练中监控泛化性能）
4. 返回 train/val/test 三个 DataLoader
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from download_mnist import ensure_mnist_downloaded

# MNIST 的经验标准化参数：像素值归一化到 [0,1] 后再减均值除标准差
# 这两个数是全网通用的（在完整训练集上统计得到）
_MNIST_MEAN = 0.1307
_MNIST_STD = 0.3081

# 加载数据时用的基础 transform：转张量 + 标准化
# 注意：不在这里做数据增强（MLP 输入要展平，增强意义不大；保持简单）
def _get_transform() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((_MNIST_MEAN,), (_MNIST_STD,))]
    )


def get_mnist_loaders(
    root: str = "data",
    batch_size: int = 128,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """返回 (train_loader, val_loader, test_loader)。

    参数
    ----
    batch_size : 每个 batch 的样本数（影响训练速度和梯度噪声）
    val_ratio : 从训练集中切多少比例做验证集
    num_workers : DataLoader 的并行加载进程数（Windows 下建议 0，避免多进程开销）
    seed : 切分验证集的随机种子，保证可复现
    """
    # 1. 确保数据已下载
    ensure_mnist_downloaded(root)

    transform = _get_transform()

    # 2. 加载完整训练集（download=False，因为 ensure_mnist_downloaded 已经搞定）
    full_train = datasets.MNIST(
        root=root, train=True, download=False, transform=transform
    )
    test_set = datasets.MNIST(
        root=root, train=False, download=False, transform=transform
    )

    # 3. 按比例切分训练集 / 验证集
    n_total = len(full_train)  # 60000
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)  # 可复现的切分
    train_set, val_set = random_split(
        full_train, [n_train, n_val], generator=generator
    )

    print(
        f"[数据] 训练集 {n_train} 张 / 验证集 {n_val} 张 / 测试集 {len(test_set)} 张"
    )

    # 4. 构建 DataLoader
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def get_raw_mnist(root: str = "data") -> tuple[torch.Tensor, torch.Tensor]:
    """返回未做标准化的原始 MNIST 测试集（用于可视化原始像素/手写样本展示）。

    返回 (images, labels)，images 形状 (N,1,28,28)，取值 [0,1]。
    """
    ensure_mnist_downloaded(root)
    raw_transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST(
        root=root, train=False, download=False, transform=raw_transform
    )
    # 一次性全部加载到内存（测试集才 10000 张，内存没问题）
    loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    images, labels = [], []
    for x, y in loader:
        images.append(x)
        labels.append(y)
    return torch.cat(images), torch.cat(labels)


class GPUBatchIterator:
    """把整个数据集预加载到 GPU 显存，迭代时直接切片。

    对 MNIST 这种小数据集（~180MB）效果显著：训练从 ~6s/epoch 降到 ~0.5s/epoch。
    原理：消除逐 batch 的 CPU→GPU 数据传输（那才是小 MLP 的真正瓶颈）。
    接口与 DataLoader 兼容（支持 for x, y in iterator），可直接替换。
    """

    def __init__(self, loader: DataLoader, batch_size: int, shuffle: bool = True,
                 device: str = "cuda"):
        # 把 DataLoader 里的所有数据一次性搬到 GPU
        xs, ys = [], []
        for x, y in loader:
            xs.append(x.to(device))
            ys.append(y.to(device))
        self.x = torch.cat(xs)
        self.y = torch.cat(ys)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(self.x)

    def __iter__(self):
        if self.shuffle:
            idx = torch.randperm(self.n, device=self.x.device)
        else:
            idx = torch.arange(self.n, device=self.x.device)
        for i in range(0, self.n, self.batch_size):
            yield self.x[idx[i:i + self.batch_size]], self.y[idx[i:i + self.batch_size]]


if __name__ == "__main__":
    # 自测：加载并打印形状
    tr, va, te = get_mnist_loaders(batch_size=64)
    xb, yb = next(iter(tr))
    print(f"一个 batch: images {xb.shape}, labels {yb.shape}")
    print(f"像素值范围: [{xb.min():.3f}, {xb.max():.3f}]")
