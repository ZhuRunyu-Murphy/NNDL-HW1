import matplotlib.pyplot as plt
import os
from model import Linear

def visualize_model_weights(model, save_dir="weight_visualizations"):
    os.makedirs(save_dir, exist_ok=True)

    for i, layer in enumerate(model.layers):
        if isinstance(layer, Linear):
            layer_idx = i // 2 + 1  # 按照你之前的命名习惯

            # ---- 权重分布直方图 ----
            plt.figure(figsize=(6, 4))
            plt.hist(layer.W.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f"Weight Distribution - Linear Layer {layer_idx}")
            plt.xlabel("Weight value")
            plt.ylabel("Frequency")
            hist_path = os.path.join(save_dir, f"weight_hist_layer{layer_idx}.png")
            plt.tight_layout()
            plt.savefig(hist_path)
            plt.close()

            # ---- 权重热力图 ----
            plt.figure(figsize=(6, 4))
            plt.imshow(layer.W, cmap="bwr", interpolation="nearest", aspect="auto", vmin=-0.1, vmax=0.1)
            plt.title(f"Weight Heatmap - Linear Layer {layer_idx}")
            plt.colorbar()
            heatmap_path = os.path.join(save_dir, f"weight_heatmap_layer{layer_idx}.png")
            plt.tight_layout()
            plt.savefig(heatmap_path)
            plt.close()

    print(f"Visualization saved to: {save_dir}")
