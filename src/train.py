"""训练循环与配置驱动的训练入口。

提供两个层次的 API：
- 低层：train_model(model, loaders, criterion, optimizer, ...) 适合自定义实验
- 高层：train_mlp(config, callback=None) 一键从配置字典训练，适合 Web 面板
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import get_mnist_loaders
from model import build_model_from_config

# 自动探测训练设备：有 CUDA 用 GPU，否则 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 支持的损失函数与优化器注册表（名字 → 构造器）
LOSS_FNS = {
    "cross_entropy": nn.CrossEntropyLoss,
    "mse": lambda: nn.MSELoss(),  # MSE 需要 one-hot 标签，下面单独处理
}

OPTIMIZERS = {
    "sgd": lambda params, lr, wd: torch.optim.SGD(params, lr=lr, weight_decay=wd),
    "momentum": lambda params, lr, wd: torch.optim.SGD(
        params, lr=lr, momentum=0.9, weight_decay=wd
    ),
    "adam": lambda params, lr, wd: torch.optim.Adam(params, lr=lr, weight_decay=wd),
}


@dataclass
class TrainingConfig:
    """一个完整的训练配置，集中管理所有可调参数。

    Web 面板和对照实验都通过构造不同的 TrainingConfig 来驱动训练。
    """

    # —— 模型结构 ——
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 128])
    activation: str = "relu"
    dropout: float = 0.0

    # —— 数据 ——
    batch_size: int = 128
    val_ratio: float = 0.1

    # —— 训练 ——
    epochs: int = 10
    lr: float = 1e-3
    optimizer: str = "adam"  # sgd / momentum / adam
    loss: str = "cross_entropy"  # cross_entropy / mse
    weight_decay: float = 0.0

    # —— 其他 ——
    seed: int = 42
    device: str = DEVICE  # 自动探测：GPU 优先，无则 CPU

    def describe(self) -> str:
        """生成一句话描述，用于图表标题。"""
        hidden_str = "-".join(str(h) for h in self.hidden_sizes)
        return (
            f"{self.activation}, [{hidden_str}], "
            f"{self.optimizer}(lr={self.lr}), {self.loss}"
        )


def _to_device(x: torch.Tensor, y: torch.Tensor, device: str):
    return x.to(device), y.to(device)


def _compute_loss(
    criterion: nn.Module,
    outputs: torch.Tensor,
    labels: torch.Tensor,
    loss_name: str,
    num_classes: int,
) -> torch.Tensor:
    """计算损失。MSE 需要把标签转成 one-hot 才能和输出对齐。"""
    if loss_name == "mse":
        # 把 (B,) 的整数标签转成 (B, num_classes) 的 one-hot float
        onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
        return criterion(outputs, onehot)
    return criterion(outputs, labels)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    loss_name: str,
    num_classes: int = 10,
) -> tuple[float, float]:
    """在给定数据集上评估，返回 (平均损失, 准确率)。"""
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for x, y in loader:
        x, y = _to_device(x, y, device)
        outputs = model(x)
        loss = _compute_loss(criterion, outputs, y, loss_name, num_classes)
        total_loss += loss.item() * x.size(0)
        total_correct += (outputs.argmax(1) == y).sum().item()
        total_n += x.size(0)
    return total_loss / total_n, total_correct / total_n


def run_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    loss_name: str = "cross_entropy",
    num_classes: int = 10,
) -> tuple[float, float, float, float]:
    """训练一个 epoch 并在验证集上评估，返回 (train_loss, train_acc, val_loss, val_acc)。

    供 Gradio 面板做"逐 epoch 流式训练"：调用方在循环里反复调用它，
    每次调用后就能刷新一次训练曲线图。
    """
    # —— 训练阶段 ——
    model.train()
    running_loss, running_correct, running_n = 0.0, 0, 0
    for x, y in train_loader:
        x, y = _to_device(x, y, device)
        outputs = model(x)
        loss = _compute_loss(criterion, outputs, y, loss_name, num_classes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        running_correct += (outputs.argmax(1) == y).sum().item()
        running_n += x.size(0)

    train_loss = running_loss / running_n
    train_acc = running_correct / running_n

    # —— 验证阶段 ——
    val_loss, val_acc = evaluate(
        model, val_loader, criterion, device, loss_name, num_classes
    )
    return train_loss, train_acc, val_loss, val_acc


def setup_training(
    config: TrainingConfig, data_root: str = "data"
) -> tuple[nn.Module, DataLoader, DataLoader, nn.Module, torch.optim.Optimizer]:
    """根据配置准备好训练所需的全部对象，返回 (model, train_loader, val_loader, criterion, optimizer)。

    与 run_one_epoch 配合，让调用方可以自己控制 epoch 循环（例如 Web 面板逐轮刷新）。
    """
    torch.manual_seed(config.seed)
    train_loader, val_loader, _ = get_mnist_loaders(
        root=data_root,
        batch_size=config.batch_size,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    # GPU 模式：把数据集整体预加载到显存，消除逐 batch 的 CPU→GPU 传输瓶颈
    # 对小 MLP 提速约 12~14 倍（~6s/epoch → ~0.5s/epoch）
    if config.device == "cuda":
        from data import GPUBatchIterator
        train_loader = GPUBatchIterator(
            train_loader, config.batch_size, shuffle=True, device="cuda"
        )
        val_loader = GPUBatchIterator(
            val_loader, config.batch_size, shuffle=False, device="cuda"
        )
    model = build_model_from_config(
        {
            "hidden_sizes": config.hidden_sizes,
            "activation": config.activation,
            "dropout": config.dropout,
        }
    ).to(config.device)
    criterion, optimizer = build_criterion_optimizer(model, config)
    return model, train_loader, val_loader, criterion, optimizer


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str = "cpu",
    loss_name: str = "cross_entropy",
    num_classes: int = 10,
    callback=None,
) -> dict:
    """执行完整训练，返回历史记录字典。

    callback(epoch, history) 会在每个 epoch 结束后被调用，
    Web 面板用它来实现训练过程中的实时曲线刷新。
    """
    history: dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": [],
    }

    for epoch in range(epochs):
        t0 = time.time()
        # 训练一个 epoch + 验证（逻辑统一走 run_one_epoch）
        train_loss, train_acc, val_loss, val_acc = run_one_epoch(
            model, train_loader, val_loader, criterion, optimizer,
            device, loss_name, num_classes,
        )
        dt = time.time() - t0
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(dt)

        print(
            f"  Epoch {epoch + 1:2d}/{epochs} "
            f"| train_loss {train_loss:.4f} acc {train_acc:.4f} "
            f"| val_loss {val_loss:.4f} acc {val_acc:.4f} "
            f"| {dt:.1f}s"
        )

        # 通知外部（Web 面板）当前进度
        if callback is not None:
            callback(epoch, history)

    return history


def build_criterion_optimizer(
    model: nn.Module, config: TrainingConfig
) -> tuple[nn.Module, torch.optim.Optimizer]:
    """根据配置构造损失函数和优化器。"""
    if config.loss not in LOSS_FNS:
        raise ValueError(f"未知损失函数: {config.loss}，可选: {list(LOSS_FNS)}")
    if config.optimizer not in OPTIMIZERS:
        raise ValueError(f"未知优化器: {config.optimizer}，可选: {list(OPTIMIZERS)}")

    criterion = LOSS_FNS[config.loss]()
    optimizer = OPTIMIZERS[config.optimizer](
        model.parameters(), lr=config.lr, wd=config.weight_decay
    )
    return criterion, optimizer


def train_mlp(
    config: TrainingConfig,
    callback=None,
    data_root: str = "data",
) -> tuple[nn.Module, dict]:
    """高层入口：从配置字典一键训练，返回 (model, history)。

    供 Web 面板、对照实验、教程脚本统一调用。
    """
    torch.manual_seed(config.seed)  # 可复现

    # 1. 数据
    train_loader, val_loader, _ = get_mnist_loaders(
        root=data_root,
        batch_size=config.batch_size,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    # 2. 模型
    model = build_model_from_config(
        {
            "hidden_sizes": config.hidden_sizes,
            "activation": config.activation,
            "dropout": config.dropout,
        }
    ).to(config.device)
    print(
        f"[模型] 参数量: {model.count_parameters():,} | 配置: {config.describe()}"
    )

    # 3. 损失 + 优化器
    criterion, optimizer = build_criterion_optimizer(model, config)

    # 4. 训练
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=config.epochs,
        device=config.device,
        loss_name=config.loss,
        callback=callback,
    )

    return model, history


if __name__ == "__main__":
    # 自测：用一个轻量配置跑 2 个 epoch
    cfg = TrainingConfig(epochs=2, hidden_sizes=[64], lr=1e-3)
    model, hist = train_mlp(cfg)
    print(f"\n最终验证准确率: {hist['val_acc'][-1]:.4f}")
