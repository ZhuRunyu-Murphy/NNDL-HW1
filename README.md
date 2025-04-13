# NNDL-HW1

Neural Networks and Deep Learning 课程作业 HW1

本项目实现了一个完整的多层感知机（MLP）图像分类器，用于在 CIFAR-10 数据集上进行训练、测试与分析，涵盖了数据加载、模型构建、训练流程、测试评估、超参数搜索及可视化等模块。

---

## 📁 文件说明

| 文件名 | 说明 |
|--------|------|
| `cifar10_loader.py` | 处理并加载 CIFAR-10 数据集，划分为训练集、验证集和测试集，支持数据预处理与缓存功能。 |
| `model.py` | 定义多层感知机模型结构，支持自定义隐藏层大小与激活函数类型，支持前向传播与反向传播计算损失梯度。 |
| `train.py` | 实现模型训练逻辑，包括 SGD 优化器、交叉熵损失、L2 正则化、学习率衰减，并根据验证集准确率自动保存最优模型权重。 |
| `hyperparameter_search.py` | 实现超参数搜索功能，可调节学习率、隐藏层大小、正则化强度等参数，并记录各超参数下模型的性能表现。 |
| `test.py` | 加载训练好的模型，在测试集上评估分类准确率，并支持可视化网络参数。 |
| `plot_loss_acc.py` | 绘制训练过程中的 loss 和 accuracy 曲线，用于分析模型收敛情况与性能。 |
| `visualize_model_weights.py` | 可视化模型的权重参数，帮助理解网络学习到的特征表示。 |

---

## 📦 数据准备

运行以下命令下载并解压 CIFAR-10 数据集：

```bash
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf cifar-10-python.tar.gz


## 🚀 使用方法

- **训练模型**：

```bash
python train.py

- **测试模型**：

```bash
python test.py

- **超参搜索**：

```bash
python hyperparameter_search.py
