import numpy as np
import matplotlib.pyplot as plt


def plot_mlp_loss(losses, save_path="mlp_loss_curve.png", epochs=10):
    # 计算每个 Mini-batch 对应的 Epoch 位置
    x = np.arange(1, len(losses) + 1)
    x = x / (len(losses) / epochs)

    plt.figure(figsize=(10, 5))
    plt.plot(x, losses, linewidth=1)

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"MLP Training Loss ({epochs} Epochs)")

    plt.xticks(range(0, epochs + 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.show()