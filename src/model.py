"""可配置的多层感知机（MLP）模型定义。

设计原则：
1. 所有超参数都通过构造函数传入，方便在 Web 面板里动态调整
2. 前向传播内部处理图像展平，调用者可以直接传 (B,1,28,28) 的图像张量
3. 激活函数、dropout 都可配置，用于演示不同选择对训练的影响
"""
from __future__ import annotations

import torch
import torch.nn as nn


# 激活函数注册表：名字 → 构造函数
# 这样可以在配置里用字符串指定，也方便 Web 面板做成下拉选项
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
}


class MLP(nn.Module):
    """一个配置驱动的多层感知机。

    参数
    ----
    input_size : int
        输入特征维数（MNIST 展平后是 28*28=784）
    hidden_sizes : list[int]
        每个隐藏层的神经元个数。例如 [256, 128] 表示两个隐藏层
    output_size : int
        输出维数（分类任务等于类别数，MNIST 是 10）
    activation : str
        隐藏层激活函数名称，见 ACTIVATIONS
    dropout : float
        dropout 概率，0.0 表示关闭
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: list[int] | None = None,
        output_size: int = 10,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        if activation not in ACTIVATIONS:
            raise ValueError(
                f"未知激活函数: {activation}，可选: {list(ACTIVATIONS)}"
            )

        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            # 一个隐藏层 = 线性变换 + 激活函数 (+ 可选 dropout)
            # 这就是教程里反复出现的 f(Wx+b) 的工程实现
            layers.append(nn.Linear(prev, h))
            layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        # 输出层：只有线性变换，不加激活、不加 dropout
        # 因为后面用 nn.CrossEntropyLoss，它内部自带 log-softmax
        layers.append(nn.Linear(prev, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 支持两种输入：
        #   - 图像张量 (B, 1, 28, 28)：先展平成 (B, 784)
        #   - 已经展平的向量 (B, 784)：直接进网络
        if x.dim() == 4:
            x = x.flatten(1)
        return self.network(x)

    def count_parameters(self) -> int:
        """返回可训练参数总数（用于展示"模型有多大"）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        """返回每一层（含激活）的输出张量。

        用于"显微镜"面板：把输入逐层推进，记录每层中间结果，
        让读者直观看到数据在网络内部是如何一步步被变换的。
        """
        if x.dim() == 4:
            x = x.flatten(1)
        activations = []
        for layer in self.network:
            x = layer(x)
            activations.append(x.detach())
        return activations


def build_model_from_config(config: dict) -> MLP:
    """从一个配置字典构造模型，方便 Web 面板和训练脚本统一调用。"""
    return MLP(
        input_size=config.get("input_size", 784),
        hidden_sizes=config.get("hidden_sizes", [256, 128]),
        output_size=config.get("output_size", 10),
        activation=config.get("activation", "relu"),
        dropout=config.get("dropout", 0.0),
    )
