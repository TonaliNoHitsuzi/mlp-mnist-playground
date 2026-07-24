"""MLP 可视化函数库。

本模块提供两大类可视化：
1. 概念示意图（不需要模型/数据）—— 给教程生成静态配图，如神经元结构、激活函数曲线、
   损失曲面、梯度回传示意、过拟合曲线等。直观地讲清楚"为什么"。
2. 实测可视化（需要模型/数据/训练历史）—— 训练曲线、权重热力图、混淆矩阵、
   隐藏层激活、预测置信度等。让读者"看到"模型内部到底发生了什么。

所有函数都返回 matplotlib Figure 对象，既能保存为 PNG 供教程引用，
也能直接喂给 Gradio 的 gr.Plot 组件做交互展示。
"""
from __future__ import annotations

import sys

# Windows 控制台默认 GBK 编码，遇到 ✓/✗ 等符号会崩溃，强制 UTF-8 输出
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

# 用 Agg 后端：无 GUI 依赖，适合保存图片 + 在 Web 服务里渲染
matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

# ============================================================
# 全局样式配置：中文字体 + 统一配色
# ============================================================
# Windows 中文 Sans 字体，SimHei 作兜底
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 负号显示
plt.rcParams["figure.dpi"] = 150  # Gradio 动态图清晰度
plt.rcParams["savefig.dpi"] = 200  # 存盘图清晰度

# 统一配色（训练/验证两条线，直观区分）
COLOR_TRAIN = "#2563eb"  # 蓝
COLOR_VAL = "#dc2626"  # 红
# 多条曲线对比用的调色板
PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]


# ============================================================
# A. 概念示意图（教程配图，不依赖模型）
# ============================================================
def plot_activations() -> plt.Figure:
    """三种激活函数曲线对比：ReLU / Sigmoid / Tanh。"""
    x = np.linspace(-6, 6, 400)
    relu = np.maximum(0, x)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, relu, color=PALETTE[0], linewidth=2.5, label="ReLU: max(0, x)")
    ax.plot(x, sigmoid, color=PALETTE[1], linewidth=2.5, label=r"Sigmoid: $\frac{1}{1+e^{-x}}$")
    ax.plot(x, tanh, color=PALETTE[2], linewidth=2.5, label=r"Tanh: $\frac{e^x - e^{-x}}{e^x + e^{-x}}$")
    ax.axhline(0, color="#888", linewidth=0.8)
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_title("三种常用激活函数对比", fontsize=13)
    ax.set_xlabel("输入 x")
    ax.set_ylabel("输出 f(x)")
    ax.set_ylim(-1.3, 2.3)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    # 标注关键区域：Sigmoid 在两侧"梯度消失"
    ax.annotate(
        "Sigmoid 两侧梯度→0\n(梯度消失)",
        xy=(5, 0.99),
        xytext=(2.2, 1.7),
        fontsize=9,
        color=PALETTE[1],
        arrowprops=dict(arrowstyle="->", color=PALETTE[1]),
    )
    fig.tight_layout()
    return fig


def plot_loss_functions() -> plt.Figure:
    """损失函数几何直觉：MSE (y-ŷ)² 与交叉熵 -log(p)。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # 左：MSE 损失曲线（预测值偏离真值时的损失）
    delta = np.linspace(-2, 2, 300)
    axes[0].plot(delta, delta ** 2, color=PALETTE[0], linewidth=2.5)
    axes[0].set_title("均方误差 MSE = (ŷ - y)²", fontsize=12)
    axes[0].set_xlabel("预测误差 (ŷ - y)")
    axes[0].set_ylabel("损失")
    axes[0].annotate(
        "误差越大 → 损失平方级增长",
        xy=(1.6, 2.56),
        xytext=(0.2, 3.0),
        fontsize=9,
        arrowprops=dict(arrowstyle="->"),
    )

    # 右：交叉熵 -log(p)，p 是给正确类的预测概率
    p = np.linspace(0.01, 1, 300)
    axes[1].plot(p, -np.log(p), color=PALETTE[1], linewidth=2.5)
    axes[1].set_title("交叉熵 CE = -log(p_correct)", fontsize=12)
    axes[1].set_xlabel("给正确类别预测的概率 p")
    axes[1].set_ylabel("损失")
    axes[1].set_ylim(0, 5)
    axes[1].annotate(
        "概率越小 → 损失爆炸\n(强烈惩罚'自信地错')",
        xy=(0.1, 2.3),
        xytext=(0.35, 3.8),
        fontsize=9,
        arrowprops=dict(arrowstyle="->"),
    )

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="#888", linewidth=0.8)
    fig.tight_layout()
    return fig


def plot_loss_landscape() -> plt.Figure:
    """梯度下降"小球下山"示意：二维损失曲面 + 下降轨迹。

    同时演示三种学习率：太小（龟速）、合适、太大（跨过最小值）。
    """
    # 构造一个碗状损失曲面（二次型）
    a, b = 1.0, 1.5
    f_loss = lambda x, y: a * x ** 2 + b * y ** 2
    grad = lambda x, y: np.array([2 * a * x, 2 * b * y])

    # 梯度下降轨迹（三种学习率）
    def descend(lr, steps, start=(2.5, 2.2)):
        path = [np.array(start)]
        for _ in range(steps):
            g = grad(*path[-1])
            path.append(path[-1] - lr * g)
        return np.array(path)

    path_small = descend(0.05, 8)  # 太小：步子慢
    path_good = descend(0.2, 8)  # 合适
    path_large = descend(0.7, 6)  # 太大：震荡

    fig, ax = plt.subplots(figsize=(7, 5.5))
    # 等高线
    gx = np.linspace(-3, 3, 100)
    gy = np.linspace(-2.5, 2.5, 100)
    GX, GY = np.meshgrid(gx, gy)
    GZ = f_loss(GX, GY)
    ax.contour(GX, GY, GZ, levels=15, cmap="Blues", alpha=0.6)
    # 最小值点
    ax.plot(0, 0, "k*", markersize=18, label="最小值")
    # 三条轨迹
    for path, label, color, ls in [
        (path_small, "学习率太小 (龟速)", PALETTE[1], "-"),
        (path_good, "学习率合适", PALETTE[2], "-"),
        (path_large, "学习率太大 (震荡)", PALETTE[3], "--"),
    ]:
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2,
                linestyle=ls, marker="o", markersize=5, label=label)
    ax.set_title("梯度下降：学习率的影响", fontsize=13)
    ax.set_xlabel("参数 $w_1$")
    ax.set_ylabel("参数 $w_2$")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2.5, 2.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_overfitting_demo() -> plt.Figure:
    """过拟合示意：训练损失持续下降，验证损失先降后升。"""
    epochs = np.arange(1, 31)
    # 构造典型曲线
    train_loss = 2.0 * np.exp(-0.25 * epochs) + 0.05 + np.random.RandomState(1).randn(30) * 0.01
    val_loss = 2.0 * np.exp(-0.2 * epochs) + 0.15
    val_loss[12:] += np.linspace(0, 0.35, 18)  # 第 12 轮后开始上升
    val_loss += np.random.RandomState(2).randn(30) * 0.015

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(epochs, train_loss, color=COLOR_TRAIN, linewidth=2.2, label="训练损失")
    ax.plot(epochs, val_loss, color=COLOR_VAL, linewidth=2.2, label="验证损失")
    # 标注早停点
    best = np.argmin(val_loss)
    ax.axvline(best + 1, color="#16a34a", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.annotate(
        f"应该在这里早停\n(第 {best + 1} 轮)",
        xy=(best + 1, val_loss[best]),
        xytext=(best + 6, val_loss[best] + 0.25),
        fontsize=9,
        color="#16a34a",
        arrowprops=dict(arrowstyle="->", color="#16a34a"),
    )
    # 标注过拟合区域
    ax.axvspan(13, 30, alpha=0.08, color=COLOR_VAL)
    ax.text(21, 1.7, "过拟合区\n(开始背答案)", fontsize=9,
            color=COLOR_VAL, ha="center")
    ax.set_title("过拟合：训练损失还在降，验证损失却上升了", fontsize=12)
    ax.set_xlabel("训练轮次 (epoch)")
    ax.set_ylabel("损失")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_neuron_diagram() -> plt.Figure:
    """单个神经元结构示意图：输入 → 加权求和 → 激活 → 输出。"""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # 三个输入节点
    for i, y in enumerate([3.7, 2.5, 1.3], 1):
        ax.add_patch(mpatches.Circle((1, y), 0.28, color="#dbeafe", ec=PALETTE[0], zorder=3))
        ax.text(1, y, f"$x_{i}$", ha="center", va="center", fontsize=12, zorder=4)
        # 连线到求和节点
        arrow = FancyArrowPatch((1.28, y), (3.6, 2.5), arrowstyle="->",
                                mutation_scale=14, color="#64748b", linewidth=1.3)
        ax.add_patch(arrow)
        ax.text(2.3, (y + 2.5) / 2 + 0.15, f"$w_{i}$", fontsize=10, color=PALETTE[1])

    # 求和节点
    ax.add_patch(mpatches.Circle((4, 2.5), 0.45, color="#fef3c7", ec="#d97706", zorder=3))
    ax.text(4, 2.5, "Σ", ha="center", va="center", fontsize=18, zorder=4)
    # 偏置标注
    ax.text(4, 1.6, "+ b", ha="center", fontsize=11, color="#d97706")

    # 到激活函数
    arrow = FancyArrowPatch((4.45, 2.5), (5.9, 2.5), arrowstyle="->",
                            mutation_scale=16, color="#64748b", linewidth=1.5)
    ax.add_patch(arrow)

    # 激活函数节点
    ax.add_patch(mpatches.Circle((6.3, 2.5), 0.45, color="#dcfce7", ec=PALETTE[2], zorder=3))
    ax.text(6.3, 2.5, "f(·)", ha="center", va="center", fontsize=12, zorder=4)
    ax.text(6.3, 1.6, "激活函数", ha="center", fontsize=9, color=PALETTE[2])

    # 到输出
    arrow = FancyArrowPatch((6.75, 2.5), (8.2, 2.5), arrowstyle="->",
                            mutation_scale=16, color="#64748b", linewidth=1.5)
    ax.add_patch(arrow)

    # 输出
    ax.add_patch(mpatches.Circle((8.6, 2.5), 0.32, color="#fce7f3", ec=PALETTE[3], zorder=3))
    ax.text(8.6, 2.5, "y", ha="center", va="center", fontsize=13, zorder=4)

    # 顶部公式
    ax.text(5, 4.5, r"$y = f(w_1 x_1 + w_2 x_2 + w_3 x_3 + b)$",
            ha="center", fontsize=14)
    ax.set_title("一个神经元 = 加权求和 + 非线性激活", fontsize=13, pad=10)
    return fig


def plot_mlp_structure() -> plt.Figure:
    """MLP 层级结构示意图（输入层 → 隐藏层 → 输出层）。"""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # 各层节点数（输入层画少量代表）
    layers = {"输入层\n(784 像素)": (1.5, 6, "#dbeafe", PALETTE[0]),
              "隐藏层1\n(256)": (4.5, 6, "#fef3c7", "#d97706"),
              "隐藏层2\n(128)": (7, 6, "#fef3c7", "#d97706"),
              "输出层\n(10 类)": (9.3, 4, "#fce7f3", PALETTE[3])}

    positions = {}
    for name, (xpos, n, fc, ec) in layers.items():
        ys = np.linspace(8.5, 1.5, n) if n > 2 else [5]
        if n > 2:
            ys = ys[:6]  # 最多画 6 个代表
        positions[xpos] = ys
        for y in ys:
            ax.add_patch(mpatches.Circle((xpos, y), 0.22, color=fc, ec=ec, zorder=3))
        # 层标签
        ax.text(xpos, 0.5, name, ha="center", fontsize=10, color=ec)

    # 连线（相邻层全连接，但只画稀疏的几条避免太乱）
    x_positions = [1.5, 4.5, 7, 9.3]
    for i in range(len(x_positions) - 1):
        x0 = x_positions[i]
        x1 = x_positions[i + 1]
        for y0 in positions[x0]:
            for y1 in positions[x1]:
                if np.random.RandomState(i).rand() < 0.5:  # 稀疏化
                    ax.plot([x0 + 0.22, x1 - 0.22], [y0, y1],
                            color="#94a3b8", linewidth=0.4, alpha=0.5, zorder=1)

    # 说明文字
    ax.text(4.5, 9.4, "MLP = 多层全连接网络（每条连线一个权重）",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(4.5, 0.05, "数据从左流向右（前向传播），每层提炼一次特征",
            ha="center", fontsize=9, color="#64748b")
    return fig


def plot_gradient_flow() -> plt.Figure:
    """梯度反向传播示意：误差信号从输出层反向流回输入层（责任分摊）。"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # 三个层（正向：左→右）
    layer_xs = [2, 5, 8]
    labels = ["隐藏层1", "隐藏层2", "输出层"]
    for x, lbl in zip(layer_xs, labels):
        ax.add_patch(mpatches.Circle((x, 4), 0.6, color="#dbeafe",
                                     ec=PALETTE[0], zorder=3))
        ax.text(x, 4, lbl, ha="center", va="center", fontsize=10, zorder=4)

    # 正向传播箭头（上方）
    for i in range(2):
        ax.annotate("", xy=(layer_xs[i + 1] - 0.7, 4.4), xytext=(layer_xs[i] + 0.7, 4.4),
                    arrowprops=dict(arrowstyle="->", color=PALETTE[0], linewidth=2))
    ax.text(5, 5.2, "前向传播：输入 → 输出", ha="center",
            color=PALETTE[0], fontsize=11, fontweight="bold")

    # 反向传播箭头（下方）
    for i in range(2):
        ax.annotate("", xy=(layer_xs[i] + 0.7, 3.0), xytext=(layer_xs[i + 1] - 0.7, 3.0),
                    arrowprops=dict(arrowstyle="->", color=COLOR_VAL, linewidth=2))
    ax.text(5, 1.8, "反向传播：损失 → 逐层算梯度（责任分摊）", ha="center",
            color=COLOR_VAL, fontsize=11, fontweight="bold")

    # 损失
    ax.add_patch(mpatches.FancyBboxPatch((9, 2.6), 0.9, 0.8,
                                         boxstyle="round,pad=0.1",
                                         color="#fee2e2", ec=COLOR_VAL, zorder=3))
    ax.text(9.45, 3.0, "损失\nL", ha="center", va="center", fontsize=10, zorder=4)

    ax.set_title("反向传播：误差信号沿网络反向流回", fontsize=13, pad=8)
    return fig


def plot_decision_boundary() -> plt.Figure:
    """单层 vs 多层决策边界对比（为什么需要多层 + 非线性）。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # 构造一个"螺旋"或"月牙"状的非线性可分数据
    rng = np.random.RandomState(42)
    theta = np.linspace(0, 2 * np.pi, 200)
    # 类0：内圈
    c0 = np.c_[1.5 * np.cos(theta) + rng.randn(200) * 0.2,
               1.5 * np.sin(theta) + rng.randn(200) * 0.2]
    # 类1：外圈
    c1 = np.c_[3.5 * np.cos(theta) + rng.randn(200) * 0.2,
               3.5 * np.sin(theta) + rng.randn(200) * 0.2]
    X = np.vstack([c0, c1])
    y = np.array([0] * 200 + [1] * 200)

    # 网格用于画决策边界
    xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))

    # 左：线性分类器（到中心的距离阈值，等效于一条圆环边界但用线性近似）
    # 用简化演示：线性 = 无法分开同心圆，画一条直线
    axes[0].scatter(c0[:, 0], c0[:, 1], color=PALETTE[0], s=10, alpha=0.6, label="类0")
    axes[0].scatter(c1[:, 0], c1[:, 1], color=PALETTE[1], s=10, alpha=0.6, label="类1")
    axes[0].plot([-5, 5], [0, 0], "k--", linewidth=1.5, label="线性边界（必然切错）")
    axes[0].set_title("单层线性模型：一条直线切不开同心圆", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(-5, 5)
    axes[0].set_ylim(-5, 5)

    # 右：非线性（用一个简单的径向判定，模拟 MLP 学到的曲线边界）
    Z = np.sqrt(xx ** 2 + yy ** 2)  # 到原点距离
    axes[1].contourf(xx, yy, Z, levels=[0, 2.5, 10], colors=[PALETTE[0], PALETTE[1]], alpha=0.2)
    axes[1].contour(xx, yy, Z, levels=[2.5], colors="k", linewidths=1.5)
    axes[1].scatter(c0[:, 0], c0[:, 1], color=PALETTE[0], s=10, alpha=0.6, label="类0")
    axes[1].scatter(c1[:, 0], c1[:, 1], color=PALETTE[1], s=10, alpha=0.6, label="类1")
    axes[1].set_title("多层 + 非线性：能拟合任意曲线边界", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(-5, 5)
    axes[1].set_ylim(-5, 5)

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ============================================================
# B. 数据可视化
# ============================================================
def plot_sample_digits(
    images: torch.Tensor, labels: torch.Tensor, n: int = 20, title: str | None = None
) -> plt.Figure:
    """展示若干手写数字样本。images: (N,1,28,28) 或 (N,28,28)。"""
    if images.dim() == 4:
        images = images.squeeze(1)
    images = images.detach().cpu()
    labels = labels.detach().cpu()

    cols = 10
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.2))
    if rows == 1:
        axes = axes[np.newaxis, :] if cols > 1 else np.array([[axes]])
    axes = np.atleast_2d(axes)

    for idx in range(rows * cols):
        r, c = idx // cols, idx % cols
        ax = axes[r][c]
        if idx < n:
            ax.imshow(images[idx], cmap="gray_r")
            ax.set_title(str(labels[idx].item()), fontsize=9)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


# ============================================================
# C. 训练过程可视化
# ============================================================
def plot_training_curves(history: dict, title: str = "训练曲线") -> plt.Figure:
    """单次训练的 loss / accuracy 双图。"""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # 损失曲线
    ax1.plot(epochs, history["train_loss"], color=COLOR_TRAIN, marker="o",
             markersize=4, label="训练损失")
    ax1.plot(epochs, history["val_loss"], color=COLOR_VAL, marker="s",
             markersize=4, label="验证损失")
    ax1.set_title("损失 (Loss)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("损失")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(epochs, history["train_acc"], color=COLOR_TRAIN, marker="o",
             markersize=4, label="训练准确率")
    ax2.plot(epochs, history["val_acc"], color=COLOR_VAL, marker="s",
             markersize=4, label="验证准确率")
    ax2.set_title("准确率 (Accuracy)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("准确率")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_multi_training_curves(
    histories: list[dict], labels: list[str], metric: str = "val_acc", title: str | None = None
) -> plt.Figure:
    """多组训练历史的对比图（参数对照实验室的核心图）。

    metric: 'val_acc' / 'val_loss' / 'train_acc' / 'train_loss'
    """
    metric_names = {
        "val_acc": "验证准确率",
        "val_loss": "验证损失",
        "train_acc": "训练准确率",
        "train_loss": "训练损失",
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for hist, lbl, color in zip(histories, labels, PALETTE):
        epochs = np.arange(1, len(hist[metric]) + 1)
        ax.plot(epochs, hist[metric], color=color, marker="o",
                markersize=4, linewidth=2, label=lbl)
    ax.set_title(title or f"不同配置的{metric_names.get(metric, metric)}对比", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_names.get(metric, metric))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ============================================================
# D. 模型内部可视化
# ============================================================
def plot_weight_heatmaps(model: torch.nn.Module, n_neurons: int = 16) -> plt.Figure:
    """把第一层权重可视化成 28×28 图像。

    这是理解 MLP "学到了什么" 最直观的方式：每个隐藏神经元对应一张
    28×28 的权重图，亮区表示它"关注"图像的哪些像素。
    """
    # 取第一层 Linear 的权重 (out_features, 784)
    first_linear = None
    for m in model.network:
        if isinstance(m, torch.nn.Linear):
            first_linear = m
            break
    if first_linear is None:
        raise ValueError("找不到 Linear 层")

    weights = first_linear.weight.data.detach().cpu()  # (n_hidden, 784)
    n_total = weights.shape[0]
    n_show = min(n_neurons, n_total)

    cols = 8
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.9, rows * 1.0))
    axes = np.atleast_2d(axes)

    for idx in range(rows * cols):
        r, c = idx // cols, idx % cols
        ax = axes[r][c]
        if idx < n_show:
            w = weights[idx].reshape(28, 28)
            ax.imshow(w, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
            ax.set_title(f"#{idx}", fontsize=8)
        ax.axis("off")

    fig.suptitle(
        f"第一层 {n_show}/{n_total} 个神经元的权重（蓝=正权重，红=负权重）",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def plot_hidden_activations(
    model: torch.nn.Module, image: torch.Tensor, true_label: int | None = None
) -> plt.Figure:
    """展示一张图片在网络内部每一层的激活情况。"""
    model.eval()
    acts = model.get_layer_activations(image.unsqueeze(0) if image.dim() == 3 else image)

    # 收集所有需要画的项：输入 + 每层激活
    items = []
    # 输入图像
    inp = image.detach().cpu()
    if inp.dim() == 3:
        inp = inp.squeeze(0)
    items.append(("输入图像", inp, "gray_r"))
    # 逐层激活（Linear 和 Dropout 的输出跳过，只看 Linear 后面的激活，更有意义）
    for i, act in enumerate(acts):
        name = model.network[i].__class__.__name__
        if name in ("Linear",) and i + 1 < len(acts):
            # Linear 层输出，画成条形
            items.append((f"Linear{i // 3 + 1}", act.squeeze(0).cpu(), "bar"))
        elif name in ("ReLU", "Sigmoid", "Tanh", "LeakyReLU", "ELU"):
            # 激活后输出，画成条形
            items.append((f"{name}", act.squeeze(0).cpu(), "bar"))

    n = len(items)
    # 只保留 Linear 层输出 + 最后输出层，减少冗余
    bar_items = [it for it in items if it[2] == "bar"]
    # 选取关键层：每个 Linear 输出 + 最后的输出层
    fig = plt.figure(figsize=(13, 4.5))
    # 第一格画输入图
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1.2] + [1] * n)
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(inp, cmap="gray_r")
    lbl_txt = f"真值: {true_label}" if true_label is not None else ""
    ax0.set_title(f"输入\n{lbl_txt}", fontsize=10)
    ax0.axis("off")

    # 每层激活画条形
    for idx, (name, act, _) in enumerate(items):
        ax = fig.add_subplot(gs[idx + 1])
        act_np = act.numpy().flatten()
        colors = [PALETTE[0] if v >= 0 else PALETTE[1] for v in act_np]
        ax.bar(range(len(act_np)), act_np, color=colors, width=1.0)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        if len(act_np) > 32:
            ax.set_yticks([])
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("一张图片在每一层的激活（蓝=正/兴奋，红=负/抑制）", fontsize=12)
    fig.tight_layout()
    return fig


# ============================================================
# E. 评估可视化
# ============================================================
@torch.no_grad()
def plot_confusion_matrix(
    model: torch.nn.Module, loader, device: str = "cpu"
) -> plt.Figure:
    """在测试集上画 10×10 混淆矩阵。"""
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        all_preds.append(out.argmax(1).cpu())
        all_labels.append(y)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    # 构造 10×10 混淆矩阵
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(labels, preds):
        cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    # 在格子里写数字
    for i in range(10):
        for j in range(10):
            color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=8, color=color)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("混淆矩阵（对角线越亮 = 预测越准）", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_prediction_confidence(
    outputs: torch.Tensor, true_label: int | None = None
) -> plt.Figure:
    """画出模型对 10 个类别的预测置信度（softmax 概率）柱状图。"""
    if outputs.dim() == 1:
        outputs = outputs.unsqueeze(0)
    probs = torch.softmax(outputs, dim=1).squeeze(0).detach().cpu().numpy()
    pred = int(np.argmax(probs))

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [PALETTE[0]] * 10
    colors[pred] = PALETTE[2]  # 预测类用绿色
    if true_label is not None:
        colors[true_label] = PALETTE[1]  # 真值类用红色
    ax.bar(range(10), probs, color=colors, edgecolor="#333", linewidth=0.5)
    ax.set_xticks(range(10))
    ax.set_xlabel("数字类别")
    ax.set_ylabel("预测概率")
    ax.set_ylim(0, 1.05)
    title = f"预测结果: {pred}"
    if true_label is not None:
        title += f"（真值: {true_label}）" + (" [正确]" if pred == true_label else " [错误]")
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


@torch.no_grad()
def plot_misclassified(
    model: torch.nn.Module, loader, n: int = 16, device: str = "cpu"
) -> plt.Figure:
    """挑出并展示被错分的样本（看模型到底错在哪里）。"""
    model.eval()
    wrong_imgs, wrong_labels, wrong_preds = [], [], []
    for x, y in loader:
        xb = x.to(device)
        out = model(xb)
        preds = out.argmax(1).cpu()
        mask = preds != y
        for i in range(x.size(0)):
            if mask[i]:
                wrong_imgs.append(x[i])
                wrong_labels.append(y[i].item())
                wrong_preds.append(preds[i].item())
        if len(wrong_imgs) >= n:
            break

    if not wrong_imgs:
        # 没有错分样本：画个提示
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "完美！没有找到错分样本", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
        ax.axis("off")
        return fig

    wrong_imgs = wrong_imgs[:n]
    cols = 8
    rows = (len(wrong_imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.4))
    axes = np.atleast_2d(axes)
    for idx in range(rows * cols):
        r, c = idx // cols, idx % cols
        ax = axes[r][c]
        if idx < len(wrong_imgs):
            img = wrong_imgs[idx].squeeze().cpu()
            ax.imshow(img, cmap="gray_r")
            ax.set_title(
                f"真:{wrong_labels[idx]}→预:{wrong_preds[idx]}",
                fontsize=8, color=COLOR_VAL,
            )
        ax.axis("off")
    fig.suptitle(f"错分样本（共展示 {len(wrong_imgs)} 个）", fontsize=12)
    fig.tight_layout()
    return fig


# ============================================================
# 一键生成所有教程静态配图
# ============================================================
def generate_all_concept_figures(outdir: str = "figures") -> None:
    """生成所有不依赖模型的概念示意图，保存到 outdir。"""
    from pathlib import Path

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    generators = {
        "activations.png": plot_activations,
        "loss_functions.png": plot_loss_functions,
        "loss_landscape.png": plot_loss_landscape,
        "overfitting.png": plot_overfitting_demo,
        "neuron.png": plot_neuron_diagram,
        "mlp_structure.png": plot_mlp_structure,
        "grad_flow.png": plot_gradient_flow,
        "decision_boundary.png": plot_decision_boundary,
    }
    for fname, fn in generators.items():
        print(f"生成 {fname} ...", flush=True)
        fig = fn()
        fig.savefig(out / fname, bbox_inches="tight")  # savefig.dpi=200 已在 rcParams 设好
        plt.close(fig)  # 及时释放，避免内存累积
    print(f"✓ 全部概念图已保存到 {out}", flush=True)


if __name__ == "__main__":
    # 命令行运行：生成所有概念示意图
    generate_all_concept_figures("figures")
