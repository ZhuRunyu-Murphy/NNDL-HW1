import matplotlib.pyplot as plt

def plot_loss_acc(train_loss, val_loss, train_acc, val_acc, save_path='training_plot.png'):
    epochs = list(range(1, len(train_loss) + 1))

    # 创建loss曲线图
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss', color='orange')
    plt.plot(epochs, val_loss, label='Validation Loss', color='gold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 创建accuracy曲线图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Accuracy', color='orange')
    plt.plot(epochs, val_acc, label='Validation Accuracy', color='gold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path, dpi=300)  # dpi 可调，默认 300 适合打印和展示

    # 显示图像
    plt.show()
