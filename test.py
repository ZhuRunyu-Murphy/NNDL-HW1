import numpy as np
import pickle
from model import MLP
from cifar10_loader import CIFAR10Loader
from visualize_model_weights import visualize_model_weights

# 加载模型
with open("best_model.pkl", 'rb') as f:
    model = pickle.load(f)

# 数据加载
data_dir = './cifar-10-batches-py'  # 替换为实际路径
cifar10_loader = CIFAR10Loader(data_dir=data_dir, val_ratio=0.1, batch_size=256)
X_train, Y_train = cifar10_loader.get_train_set()
X_val, Y_val = cifar10_loader.get_val_set()
X_test, Y_test = cifar10_loader.get_test_set()

def accuracy(preds, labels):
    return np.mean(preds == labels)

train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)
train_acc = accuracy(train_pred, Y_train)
val_acc = accuracy(val_pred, Y_val)
test_acc = accuracy(test_pred, Y_test)
print('Train Accuracy:', train_acc)
print('Validation Accuracy:', val_acc)
print('Test Accuracy:', test_acc)

visualize_model_weights(model)