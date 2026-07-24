"""MLP 交互式演示面板（Gradio）。

四个 Tab：
  1. 训练台    —— 调超参数，实时看训练曲线，训练完的模型供其他 Tab 使用
  2. 预测台    —— 手写数字 → 实时预测 + 置信度 + 隐藏层激活
  3. 显微镜    —— 拆开模型看内部：权重热力图 / 混淆矩阵 / 错分样本
  4. 参数对照实验室 —— 两组配置并排训练，直观对比不同选择的影响（演示模型的灵魂）

启动：python app.py  然后浏览器打开 http://127.0.0.1:7860
"""
from __future__ import annotations

import sys
from pathlib import Path

# Windows 控制台 UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 把 src/ 加入模块搜索路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from data import get_mnist_loaders, get_raw_mnist  # noqa: E402
from model import build_model_from_config, MLP  # noqa: E402
from train import (  # noqa: E402
    TrainingConfig,
    setup_training,
    run_one_epoch,
    build_criterion_optimizer,
    evaluate,
    DEVICE,
)
from visualize import (  # noqa: E402
    plot_training_curves,
    plot_multi_training_curves,
    plot_weight_heatmaps,
    plot_hidden_activations,
    plot_confusion_matrix,
    plot_prediction_confidence,
    plot_misclassified,
    plot_sample_digits,
)

# ============================================================
# 全局缓存：测试集 loader（只加载一次）、默认模型
# ============================================================
_CACHE: dict = {"test_loader": None, "raw_images": None, "raw_labels": None}


def get_test_loader():
    if _CACHE["test_loader"] is None:
        _, _, test_loader = get_mnist_loaders(batch_size=128)
        _CACHE["test_loader"] = test_loader
    return _CACHE["test_loader"]


def get_raw_test_images():
    if _CACHE["raw_images"] is None:
        imgs, labels = get_raw_mnist()
        _CACHE["raw_images"] = imgs
        _CACHE["raw_labels"] = labels
    return _CACHE["raw_images"], _CACHE["raw_labels"]


def try_load_default_model() -> MLP | None:
    """尝试加载预训练的默认模型，让"预测台/显微镜"开箱即用。"""
    ckpt = Path("checkpoints/default_mlp.pt")
    if not ckpt.exists():
        return None
    try:
        model = build_model_from_config({"hidden_sizes": [256, 128], "activation": "relu"})
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.eval()
        return model
    except Exception as e:
        print(f"[警告] 加载默认模型失败: {e}")
        return None


class _ModelState:
    """模型包装器：避免 Gradio 把 nn.Module 当作 callable 调用。

    Gradio 的 gr.State 在初始化时，如果 value 是 callable 会尝试调用它获取初始值，
    而 nn.Module 本身就是 callable（会触发 forward），所以必须包一层。
    """

    def __init__(self, model: MLP | None = None):
        self.model = model


def _get_model(state) -> MLP | None:
    """从 gr.State 里取出真正的模型，没有则尝试加载默认模型。"""
    if state is None:
        return try_load_default_model()
    if isinstance(state, _ModelState):
        if state.model is not None:
            return state.model
        return try_load_default_model()
    # 兼容直接传入的 model 对象
    return state


# ============================================================
# 解析"隐藏层"字符串（如 "256,128"）→ list[int]
# ============================================================
def parse_hidden_sizes(s: str) -> list[int]:
    """把 '256,128' 或 '256' 解析成 [256,128]。"""
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return [int(p) for p in parts]


# ============================================================
# Tab 1：训练台
# ============================================================
def train_tab_handler(
    hidden_sizes_str, activation, dropout, batch_size, epochs, lr, optimizer, loss_fn, progress=gr.Progress(),
):
    """生成器函数：逐 epoch 训练，每次 yield 刷新训练曲线图和状态。"""
    plt.close("all")  # 清理上一轮残留图形，释放内存
    config = TrainingConfig(
        hidden_sizes=parse_hidden_sizes(hidden_sizes_str),
        activation=activation,
        dropout=float(dropout),
        batch_size=int(batch_size),
        epochs=int(epochs),
        lr=float(lr),
        optimizer=optimizer,
        loss=loss_fn,
    )

    progress(0, desc="准备数据和模型...")
    model, train_loader, val_loader, criterion, optimizer_obj = setup_training(config)
    n_params = model.count_parameters()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time": []}

    for epoch in range(config.epochs):
        progress((epoch + 1) / config.epochs, desc=f"训练第 {epoch + 1}/{config.epochs} 轮...")
        tl, ta, vl, va = run_one_epoch(
            model, train_loader, val_loader, criterion, optimizer_obj,
            device=DEVICE, loss_name=config.loss,
        )
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)

        fig = plot_training_curves(history, title=f"训练中 · 第 {epoch + 1}/{config.epochs} 轮")
        status = (
            f"✅ 第 {epoch + 1}/{config.epochs} 轮完成\n"
            f"训练损失 {tl:.4f} / 准确率 {ta:.4f}\n"
            f"验证损失 {vl:.4f} / 准确率 {va:.4f}"
        )
        yield fig, status, _ModelState(model)

    final_fig = plot_training_curves(history, title=f"训练完成 · 最终验证准确率 {history['val_acc'][-1]:.4f}")
    final_status = (
        f"🎉 训练完成！\n参数量 {n_params:,}\n"
        f"最终验证准确率: {history['val_acc'][-1]:.4f}\n"
        f"模型已就绪 → 可在「预测台」「显微镜」里使用"
    )
    yield final_fig, final_status, _ModelState(model)


# ============================================================
# Tab 2：预测台
# ============================================================
def preprocess_sketchpad(img_input) -> torch.Tensor | None:
    """把 Sketchpad 的输出转成模型可接受的标准化 28×28 张量。

    Sketchpad 返回可能是：
      - numpy 数组 (H,W,4) RGBA
      - dict: {'background':..., 'layers':[...], 'composite':...}
    """
    if img_input is None:
        return None

    # 提取像素数据
    if isinstance(img_input, dict):
        img = img_input.get("composite", img_input.get("background", None))
        if img is None and img_input.get("layers"):
            img = img_input["layers"][0]
        if img is None:
            return None
    else:
        img = img_input

    img = np.asarray(img)
    if img.size == 0:
        return None

    # 转灰度
    if img.ndim == 3:
        if img.shape[2] == 4:  # RGBA
            # 用 alpha 通道或亮度
            if img[:, :, 3].max() > 0:
                gray = img[:, :, 3].astype(np.float32)  # 画笔痕迹在 alpha 上
            else:
                gray = img[:, :, :3].mean(axis=2)
        else:
            gray = img[:, :, :3].mean(axis=2)
    else:
        gray = img.astype(np.float32)

    # Sketchpad 默认黑笔白底；MNIST 是白字黑底 → 反转
    # 判断背景：如果均值偏高（白底），则需要反转
    if gray.mean() > 127:
        gray = 255 - gray

    # 裁剪到笔迹包围盒，居中
    pil = Image.fromarray(gray.astype(np.uint8))
    bbox = pil.getbbox()
    if bbox is None:  # 空白画板
        return None
    cropped = pil.crop(bbox)
    # 等比缩放到 20×20，再 padding 到 28×28（与 MNIST 官方一致）
    cropped.thumbnail((20, 20), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    offset = ((28 - cropped.size[0]) // 2, (28 - cropped.size[1]) // 2)
    canvas.paste(cropped, offset)
    arr = np.array(canvas, dtype=np.float32) / 255.0
    # 标准化（与训练一致）
    arr = (arr - 0.1307) / 0.3081

    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
    return tensor


def predict_drawn(img_input, model_state):
    """处理手写画板输入，返回预测结果图 + 隐藏层激活图。"""
    plt.close("all")
    model = _get_model(model_state)
    if model is None:
        return None, None, "⚠️ 请先在「训练台」训练一个模型（或运行 `python scripts/train_default.py`）"

    x = preprocess_sketchpad(img_input)
    if x is None:
        return None, None, "请先在画板上写一个数字（0~9）"

    model.eval()
    with torch.no_grad():
        out = model(x)

    conf_fig = plot_prediction_confidence(out.squeeze(0))
    act_fig = plot_hidden_activations(model, x.squeeze(0))
    return conf_fig, act_fig, f"预测完成。模型参数量 {model.count_parameters():,}"


def predict_from_testset(idx, model_state):
    """从测试集取一张图来预测（方便对比，避免手写偏差）。"""
    plt.close("all")
    model = _get_model(model_state)
    if model is None:
        return None, None, "⚠️ 请先在「训练台」训练一个模型"

    images, labels = get_raw_test_images()
    idx = int(idx) % len(images)
    # 取原始图（未标准化）做展示，标准化后送入模型
    raw = images[idx]  # (1, 28, 28)
    label = labels[idx].item()

    model.eval()
    # 标准化并补上 batch 维 → (1, 1, 28, 28)
    x = ((raw - 0.1307) / 0.3081).unsqueeze(0)
    with torch.no_grad():
        out = model(x)

    conf_fig = plot_prediction_confidence(out.squeeze(0), true_label=label)
    act_fig = plot_hidden_activations(model, x.squeeze(0), true_label=label)
    return conf_fig, act_fig, f"测试集第 {idx} 张（真值 {label}）"


# ============================================================
# Tab 3：显微镜
# ============================================================
def microscope_handler(model_state, show_neurons):
    """展示模型内部：权重热力图 + 混淆矩阵 + 错分样本。"""
    plt.close("all")  # 清理上一轮图形
    model = _get_model(model_state)
    if model is None:
        return None, None, None, "⚠️ 请先在「训练台」训练一个模型"

    weight_fig = plot_weight_heatmaps(model, n_neurons=int(show_neurons))
    test_loader = get_test_loader()
    cm_fig = plot_confusion_matrix(model, test_loader)
    miss_fig = plot_misclassified(model, test_loader, n=16)
    info = f"模型参数量: {model.count_parameters():,}"
    return weight_fig, cm_fig, miss_fig, info


# ============================================================
# Tab 4：参数对照实验室
# ============================================================
# 预设对照实验：名字 → (两组 TrainingConfig, 说明)
def _cfg(**kw) -> TrainingConfig:
    return TrainingConfig(epochs=8, **kw)


COMPARISON_PRESETS: dict[str, dict] = {
    "Sigmoid vs ReLU（梯度消失）": {
        "configs": [
            _cfg(activation="sigmoid", hidden_sizes=[128]),
            _cfg(activation="relu", hidden_sizes=[128]),
        ],
        "labels": ["Sigmoid", "ReLU"],
        "metric": "val_acc",
        "explain": (
            "**为什么 ReLU 几乎总是赢？**\n"
            "Sigmoid 在输入绝对值稍大时梯度趋近 0，误差信号传不回去（梯度消失），"
            "网络越深越学不动。ReLU 在正数区梯度恒为 1，信号畅通无阻。"
        ),
    },
    "浅层(1层) vs 深层(3层)": {
        "configs": [
            _cfg(hidden_sizes=[128]),
            _cfg(hidden_sizes=[256, 128, 64]),
        ],
        "labels": ["单隐藏层 [128]", "三隐藏层 [256,128,64]"],
        "metric": "val_acc",
        "explain": (
            "**更深 = 更强表达能力，但更易过拟合 + 更慢。**\n"
            "注意看三层网络的训练速度和最终准确率，以及它是否比单层真的好很多。"
        ),
    },
    "窄(32) vs 宽(256) 网络": {
        "configs": [
            _cfg(hidden_sizes=[32]),
            _cfg(hidden_sizes=[256]),
        ],
        "labels": ["窄 [32]", "宽 [256]"],
        "metric": "val_acc",
        "explain": (
            "**窄网络容量小（记不住）、宽网络容量大（可能过拟合）。**\n"
            "MNIST 较简单，两者差距不会太大，但宽网络参数量是窄网络的约 8 倍。"
        ),
    },
    "MSE vs 交叉熵（损失函数）": {
        "configs": [
            _cfg(loss="mse", hidden_sizes=[128]),
            _cfg(loss="cross_entropy", hidden_sizes=[128]),
        ],
        "labels": ["MSE 损失", "交叉熵损失"],
        "metric": "val_acc",
        "explain": (
            "**分类任务为什么不用 MSE？**\n"
            "MSE 配合 softmax 在「自信地错」时梯度很小，学得慢；"
            "交叉熵给出又大又准的梯度，收敛更快、最终更准。"
        ),
    },
    "学习率对比 (0.001 / 0.01 / 0.1)": {
        "configs": [
            _cfg(lr=0.001, hidden_sizes=[128]),
            _cfg(lr=0.01, hidden_sizes=[128]),
            _cfg(lr=0.1, hidden_sizes=[128]),
        ],
        "labels": ["lr=0.001", "lr=0.01", "lr=0.1"],
        "metric": "val_acc",
        "explain": (
            "**学习率是最关键的超参数。**\n"
            "太小 → 收敛极慢；合适 → 稳步下降；太大 → 可能震荡甚至发散（loss 变 NaN）。"
            "试试把某个改成 1.0 看它会不会炸。"
        ),
    },
    "无 Dropout vs Dropout=0.5": {
        "configs": [
            _cfg(dropout=0.0, hidden_sizes=[256, 256]),
            _cfg(dropout=0.5, hidden_sizes=[256, 256]),
        ],
        "labels": ["无 Dropout", "Dropout=0.5"],
        "metric": "val_acc",
        "explain": (
            "**Dropout 随机关闭神经元，逼网络学到冗余表示，抑制过拟合。**\n"
            "对浅层小网络效果不明显；但在大网络/长训练时能有效缩小 train-val 差距。"
        ),
    },
}


def comparison_handler(preset_name, progress=gr.Progress()):
    """对照实验：并排训练多组配置，对比曲线。"""
    plt.close("all")  # 释放上一轮的 matplotlib 图形，避免内存累积
    if preset_name not in COMPARISON_PRESETS:
        return None, "请选择一个对照实验", ""
    preset = COMPARISON_PRESETS[preset_name]
    configs = preset["configs"]
    labels = preset["labels"]
    metric = preset["metric"]

    histories = []
    total = len(configs)
    for i, (cfg, lbl) in enumerate(zip(configs, labels)):
        progress((i + 1) / total, desc=f"训练配置 {i + 1}/{total}: {lbl}")
        model, train_loader, val_loader, criterion, opt = setup_training(cfg)
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time": []}
        for epoch in range(cfg.epochs):
            tl, ta, vl, va = run_one_epoch(
                model, train_loader, val_loader, criterion, opt,
                device=DEVICE, loss_name=cfg.loss,
            )
            history["train_loss"].append(tl)
            history["train_acc"].append(ta)
            history["val_loss"].append(vl)
            history["val_acc"].append(va)
        histories.append(history)
        del model, opt  # 释放本轮模型/优化器
        if DEVICE == "cuda":
            torch.cuda.empty_cache()  # 回收 GPU 显存

    fig = plot_multi_training_curves(histories, labels, metric=metric, title=preset_name)

    # 汇总最终结果
    result_lines = ["| 配置 | 最终验证准确率 |", "|------|---------------|"]
    for lbl, hist in zip(labels, histories):
        result_lines.append(f"| {lbl} | {hist['val_acc'][-1]:.4f} |")
    summary = "\n".join(result_lines)

    return fig, summary, preset["explain"]


def custom_comparison_handler(
    a_hidden, a_act, a_lr, a_opt, a_loss,
    b_hidden, b_act, b_lr, b_opt, b_loss, epochs, progress=gr.Progress(),
):
    """自定义两组配置并排训练对比。"""
    plt.close("all")  # 释放上一轮的 matplotlib 图形
    cfg_a = TrainingConfig(
        hidden_sizes=parse_hidden_sizes(a_hidden), activation=a_act,
        lr=float(a_lr), optimizer=a_opt, loss=a_loss, epochs=int(epochs),
    )
    cfg_b = TrainingConfig(
        hidden_sizes=parse_hidden_sizes(b_hidden), activation=b_act,
        lr=float(b_lr), optimizer=b_opt, loss=b_loss, epochs=int(epochs),
    )
    configs = [cfg_a, cfg_b]
    labels = [f"A:{a_act},lr={a_lr}", f"B:{b_act},lr={b_lr}"]

    histories = []
    for i, (cfg, lbl) in enumerate(zip(configs, labels)):
        progress((i + 1) / 2, desc=f"训练配置 {lbl}")
        model, train_loader, val_loader, criterion, opt = setup_training(cfg)
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time": []}
        for epoch in range(cfg.epochs):
            tl, ta, vl, va = run_one_epoch(
                model, train_loader, val_loader, criterion, opt,
                device=DEVICE, loss_name=cfg.loss,
            )
            history["train_loss"].append(tl)
            history["train_acc"].append(ta)
            history["val_loss"].append(vl)
            history["val_acc"].append(va)
        histories.append(history)
        del model, opt
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    fig = plot_multi_training_curves(histories, labels, metric="val_acc", title="自定义对照实验")
    summary = (
        f"配置A ({labels[0]}): {histories[0]['val_acc'][-1]:.4f}\n"
        f"配置B ({labels[1]}): {histories[1]['val_acc'][-1]:.4f}"
    )
    return fig, summary, ""


# ============================================================
# 构建 Gradio 界面
# ============================================================
def build_app():
    # 启动时尝试加载默认模型
    default_model = try_load_default_model()
    initial_state = _ModelState(default_model)

    with gr.Blocks(title="MLP 神经网络交互演示") as app:
        gr.Markdown(
            "# 🧠 MLP 多层感知机交互演示\n"
            "在 MNIST 手写数字识别任务上，亲手调参数、看模型内部、做对照实验。\n\n"
            "**推荐学习顺序**：训练台 → 显微镜 → 预测台 → 参数对照实验室。"
        )

        # 模型状态：训练台训练后写入，其他 Tab 共用
        model_state = gr.State(value=initial_state)

        # ============ Tab 1：训练台 ============
        with gr.Tab("🏋️ 训练台"):
            gr.Markdown("### 调整超参数，实时观察训练曲线\n训练完的模型会自动供其他 Tab 使用。")
            device_badge = gr.Markdown(
                f"> ⚡ **训练设备**：`{DEVICE}`"
                + ("（GPU 加速已启用）" if DEVICE == "cuda" else "（CPU 模式）")
            )
            with gr.Row():
                with gr.Column(scale=1):
                    hidden_in = gr.Textbox(value="256,128", label="隐藏层（逗号分隔神经元数）")
                    activation_in = gr.Dropdown(
                        ["relu", "sigmoid", "tanh", "leaky_relu", "elu"],
                        value="relu", label="激活函数",
                    )
                    dropout_in = gr.Slider(0, 0.8, value=0.0, step=0.1, label="Dropout 概率")
                    batch_in = gr.Slider(32, 512, value=128, step=32, label="Batch Size")
                    epochs_in = gr.Slider(1, 30, value=10, step=1, label="训练轮数 (Epochs)")
                    lr_in = gr.Slider(0.0001, 1.0, value=0.001, step=0.0001, label="学习率 (lr)")
                    opt_in = gr.Dropdown(["adam", "sgd", "momentum"], value="adam", label="优化器")
                    loss_in = gr.Dropdown(["cross_entropy", "mse"], value="cross_entropy", label="损失函数")
                    train_btn = gr.Button("🚀 开始训练", variant="primary")
                with gr.Column(scale=2):
                    train_status = gr.Textbox(label="训练状态", lines=4, interactive=False)
                    train_plot = gr.Plot(label="训练曲线")

            train_btn.click(
                train_tab_handler,
                inputs=[hidden_in, activation_in, dropout_in, batch_in,
                        epochs_in, lr_in, opt_in, loss_in],
                outputs=[train_plot, train_status, model_state],
            )

        # ============ Tab 2：预测台 ============
        with gr.Tab("✍️ 预测台"):
            gr.Markdown("### 手写一个数字，让模型预测\n或在测试集里挑一张图看预测结果。")
            with gr.Row():
                with gr.Column(scale=1):
                    sketch = gr.Sketchpad(
                        label="手写画板（用鼠标画一个 0~9）",
                        height=300, width=300, image_mode="L",
                    )
                    predict_btn = gr.Button("🔮 识别手写数字", variant="primary")
                    gr.Markdown("---\n**或从测试集取图：**")
                    test_idx = gr.Slider(0, 9999, value=0, step=1, label="测试集索引")
                    test_btn = gr.Button("📊 预测这张测试图")
                    model_info = gr.Textbox(label="模型信息", interactive=False)
                with gr.Column(scale=2):
                    conf_plot = gr.Plot(label="预测置信度（10 类概率）")
                    act_plot = gr.Plot(label="隐藏层激活")

            predict_btn.click(
                predict_drawn, inputs=[sketch, model_state],
                outputs=[conf_plot, act_plot, model_info],
            )
            test_btn.click(
                predict_from_testset, inputs=[test_idx, model_state],
                outputs=[conf_plot, act_plot, model_info],
            )

        # ============ Tab 3：显微镜 ============
        with gr.Tab("🔬 显微镜"):
            gr.Markdown("### 拆开模型，看看它学到了什么\n**权重热力图**：每个神经元「关注」图像的哪些区域；\n**混淆矩阵**：哪些数字最容易混淆；\n**错分样本**：模型到底错在哪里。")
            with gr.Row():
                neuron_in = gr.Slider(4, 64, value=16, step=4, label="展示多少个神经元的权重")
                micro_btn = gr.Button("🔍 分析当前模型", variant="primary")
            micro_info = gr.Textbox(label="模型信息", interactive=False)
            with gr.Row():
                weight_plot = gr.Plot(label="第一层权重热力图")
            with gr.Row():
                cm_plot = gr.Plot(label="测试集混淆矩阵")
            with gr.Row():
                miss_plot = gr.Plot(label="错分样本")

            micro_btn.click(
                microscope_handler, inputs=[model_state, neuron_in],
                outputs=[weight_plot, cm_plot, miss_plot, micro_info],
            )

        # ============ Tab 4：参数对照实验室 ============
        with gr.Tab("🎛️ 参数对照实验室"):
            gr.Markdown(
                "### 演示模型的灵魂：直观感受参数选择的影响\n"
                "选一个预设对照实验，多组配置并排训练，同图对比曲线。"
                "**这是理解「为什么 ReLU/MSE/学习率/...」最好的方式。**"
            )
            with gr.Row():
                preset_in = gr.Dropdown(
                    list(COMPARISON_PRESETS.keys()),
                    value="Sigmoid vs ReLU（梯度消失）",
                    label="选择对照实验",
                )
                run_preset_btn = gr.Button("🧪 运行对照实验", variant="primary")
            preset_plot = gr.Plot(label="对照训练曲线")
            preset_summary = gr.Textbox(label="最终结果", lines=4, interactive=False)
            preset_explain = gr.Markdown()

            run_preset_btn.click(
                comparison_handler, inputs=[preset_in],
                outputs=[preset_plot, preset_summary, preset_explain],
            )

            # 自定义对照
            gr.Markdown("---\n### 自定义两组配置对照")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**配置 A**")
                    ca_hidden = gr.Textbox(value="128", label="隐藏层")
                    ca_act = gr.Dropdown(["relu", "sigmoid", "tanh"], value="relu", label="激活")
                    ca_lr = gr.Slider(0.0001, 1.0, value=0.001, step=0.0001, label="学习率")
                    ca_opt = gr.Dropdown(["adam", "sgd", "momentum"], value="adam", label="优化器")
                    ca_loss = gr.Dropdown(["cross_entropy", "mse"], value="cross_entropy", label="损失")
                with gr.Column():
                    gr.Markdown("**配置 B**")
                    cb_hidden = gr.Textbox(value="128", label="隐藏层")
                    cb_act = gr.Dropdown(["relu", "sigmoid", "tanh"], value="sigmoid", label="激活")
                    cb_lr = gr.Slider(0.0001, 1.0, value=0.001, step=0.0001, label="学习率")
                    cb_opt = gr.Dropdown(["adam", "sgd", "momentum"], value="adam", label="优化器")
                    cb_loss = gr.Dropdown(["cross_entropy", "mse"], value="cross_entropy", label="损失")
            custom_epochs = gr.Slider(3, 20, value=8, step=1, label="训练轮数")
            custom_btn = gr.Button("⚔️ A vs B 对决", variant="primary")
            custom_plot = gr.Plot(label="自定义对照曲线")
            custom_summary = gr.Textbox(label="结果", interactive=False)

            custom_btn.click(
                custom_comparison_handler,
                inputs=[ca_hidden, ca_act, ca_lr, ca_opt, ca_loss,
                        cb_hidden, cb_act, cb_lr, cb_opt, cb_loss, custom_epochs],
                outputs=[custom_plot, custom_summary, preset_explain],
            )

        gr.Markdown(
            "---\n📚 配套教程见 `tutorials/` 目录 | 源码见 `src/` | "
            "💡 提示：CPU 训练，每轮约 5~15 秒，请耐心等待。"
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True,
               theme=gr.themes.Soft())
