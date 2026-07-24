# MLP 多层感知机 · 从原理到交互演示

> 一个为神经网络新手设计的学习项目：**原理教程 + 可调演示模型 + 交互式可视化面板**。
> 在经典 MNIST 手写数字识别任务上，把 MLP 从"知道是什么"变成"真正懂为什么"。

## 这是什么

本项目用三种形式帮你吃透 MLP（多层感知机）：

1. **三篇中文教程**（`tutorials/`）—— 从原理直觉到 PyTorch 工程实践到建模方法论
2. **纯 PyTorch 演示模型**（`src/`）—— 可配置、可训练、结构清晰
3. **Gradio 交互面板**（`app.py`）—— 调参数、看模型内部、做对照实验

## 快速开始

```bash
# 1. 安装依赖（PyTorch/numpy/matplotlib 应已就绪，只需补 gradio）
pip install -r requirements.txt

# 2. 预训练默认模型（GPU 约 15 秒 / CPU 约 1~2 分钟）—— 让"预测台/显微镜"开箱即用
python scripts/train_default.py

# 3. 启动交互面板
python app.py
# 浏览器自动打开 http://127.0.0.1:7860
```

> MNIST 数据在首次运行时自动下载（多镜像回退），之后离线可用。
> 训练设备自动探测：有 NVIDIA GPU 自动启用 CUDA 加速，无则回退 CPU。

## 交互面板四个 Tab

| Tab | 功能 | 学到什么 |
|-----|------|---------|
| 🏋️ **训练台** | 调超参数实时训练，看曲线动态刷新 | 每个超参数（lr/层数/激活/...）怎么影响训练 |
| ✍️ **预测台** | 手写数字 → 实时预测 + 置信度 + 隐藏层激活 | 模型如何把像素变成类别，内部激活长什么样 |
| 🔬 **显微镜** | 权重热力图 + 混淆矩阵 + 错分样本 | 模型学到了什么、错在哪、哪些类易混 |
| 🎛️ **参数对照实验室** | 两组配置并排训练对比（6 组预设实验） | 直观感受激活函数/损失函数/学习率/...的选择差异 |

## 教程目录（推荐阅读顺序）

| 章 | 标题 | 核心内容 |
|---|------|---------|
| 01 | [MLP 原理入门](./tutorials/01_MLP原理入门.md) | 神经元 → 多层结构 → 激活函数 → 损失 → 反向传播 → 优化器 → 过拟合，全程直觉讲解 |
| 02 | [PyTorch 工程实践要点](./tutorials/02_PyTorch工程实践要点.md) | 拆黑盒（`nn.Linear`/`backward`/`step`）+ 标准训练模板 + 调参诊断手册 |
| 03 | [建模流程与局限性](./tutorials/03_建模流程与局限性.md) | 通用建模七步法 + MLP 四大局限 → 为什么需要 CNN |

## 项目结构

```text
.
├── tutorials/                  # 三篇教程（Markdown）
│   ├── 01_MLP原理入门.md
│   ├── 02_PyTorch工程实践要点.md
│   └── 03_建模流程与局限性.md
├── src/                        # 演示模型核心代码（纯 PyTorch）
│   ├── model.py                #   可配置 MLP（层数/激活/dropout）
│   ├── data.py                 #   MNIST 加载 + 标准化 + 验证集切分
│   ├── train.py                #   训练循环（配置驱动 + 流式回调 + GPU 预加载）
│   ├── visualize.py            #   可视化函数库（10+ 种图）
│   └── download_mnist.py       #   MNIST 多镜像下载器
├── app.py                      # Gradio 交互面板（四个 Tab）
├── scripts/
│   └── train_default.py        # 预训练默认模型
├── figures/                    # 教程引用的静态配图（已纳入仓库）
├── checkpoints/                # 预训练权重（gitignore，运行 train_default.py 生成）
├── data/                       # MNIST 数据缓存（gitignore，首次运行自动下载）
├── requirements.txt
└── .gitignore
```

## 关键设计决策

- **纯 PyTorch 而非 NumPy 从零实现**：贴近工程实战，原理放在教程里讲透
- **GPU 自动加速**：有 NVIDIA 显卡时数据预加载到显存，单 epoch 从 ~6s 降到 ~0.5s；无 GPU 回退 CPU 也能跑
- **静态图 + 交互面板双轨**：教程配图保证图文一致，面板支持自由探索
- **参数对照实验室**：演示模型的灵魂——让读者亲眼看到不同选择的速度/性能差异

## 关于版本控制

仓库只包含**源码和教程**。以下为可由代码重新生成的计算结果，已通过 `.gitignore` 排除：

| 目录 | 内容 | 重新生成方式 |
|------|------|-------------|
| `data/` | MNIST 数据集（约 60MB） | 首次运行时 `src/download_mnist.py` 自动下载（多镜像回退） |
| `checkpoints/` | 预训练模型权重 | `python scripts/train_default.py`（GPU 约 10 秒） |

> `figures/` 下的 8 张概念配图虽由代码生成，但作为教程的静态引用图已纳入仓库，无需重新生成（如需重生成：`python src/visualize.py`）。

## 环境要求

### 已验证环境

| 组件 | 版本 |
|------|------|
| 操作系统 | Windows 11 |
| Python | 3.12.4 |
| PyTorch | 2.11.0+cu128（CUDA 12.8，GPU 加速） |
| torchvision | 0.26.0+cu128 |
| Gradio | 6.20.0 |
| numpy / matplotlib / scikit-learn / Pillow / requests | 见 `requirements.txt` |
| GPU | NVIDIA RTX 4060 Laptop（8GB） |

### 安装

```bash
pip install -r requirements.txt
# 有 NVIDIA GPU 建议装 CUDA 版 PyTorch 以获得 10 倍以上训练加速：
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# 无 GPU 也可用 CPU 版（pip install torch），训练稍慢但功能完全一致
```
