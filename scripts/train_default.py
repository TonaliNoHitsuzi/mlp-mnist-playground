"""预训练默认 MLP 模型，保存到 checkpoints/default_mlp.pt。

这样 Web 面板的「预测台」「显微镜」开箱即用，无需先在训练台手动训练。
命令：python scripts/train_default.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Windows UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 让脚本能 import src 下的模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from model import build_model_from_config  # noqa: E402
from train import TrainingConfig, setup_training, run_one_epoch  # noqa: E402


def main():
    # 默认配置：两层隐藏层 + ReLU + Adam，一个稳健的 baseline
    config = TrainingConfig(
        hidden_sizes=[256, 128],
        activation="relu",
        dropout=0.0,
        batch_size=128,
        epochs=12,
        lr=1e-3,
        optimizer="adam",
        loss="cross_entropy",
    )
    print(f"[预训练] 配置: {config.describe()}")
    print(f"[预训练] 开始训练（GPU 约 10 秒 / CPU 约 1~2 分钟）...")

    model, train_loader, val_loader, criterion, optimizer = setup_training(config)
    print(f"[模型] 参数量: {model.count_parameters():,}")

    best_val_acc = 0.0
    best_state = None
    for epoch in range(config.epochs):
        tl, ta, vl, va = run_one_epoch(
            model, train_loader, val_loader, criterion, optimizer,
            device=config.device, loss_name=config.loss,
        )
        print(
            f"  Epoch {epoch + 1:2d}/{config.epochs} "
            f"| train_loss {tl:.4f} acc {ta:.4f} "
            f"| val_loss {vl:.4f} acc {va:.4f}"
        )
        # 保留验证准确率最高的权重
        if va > best_val_acc:
            best_val_acc = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # 恢复最佳权重并保存
    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "default_mlp.pt"
    torch.save(model.state_dict(), out_path)
    print(f"\n[预训练] ✓ 最佳验证准确率: {best_val_acc:.4f}")
    print(f"[预训练] ✓ 已保存到 {out_path}")
    print("[预训练] 现在 app.py 的预测台/显微镜可直接使用。")


if __name__ == "__main__":
    main()
