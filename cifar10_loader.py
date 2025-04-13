import pickle
import numpy as np
import os

class CIFAR10Loader:
    def __init__(self, data_dir, val_ratio=0.1, batch_size=128):
        """
        初始化数据加载器

        参数:
        - data_dir: CIFAR-10 数据集目录
        - val_ratio: 验证集所占训练集比例
        - batch_size: 每个 batch 的大小
        """
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.seed = 42
        self.label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']
        
        self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test = self._load_and_split()

    def _load_cifar_batch(self, filename):
        """
        加载 CIFAR-10 中单个 batch 文件

        返回:
        - X: 图像数据，形状为 [10000, 3072]
        - Y: 标签，形状为 [10000]
        """
        with open(filename, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
            X = data_dict[b'data'].astype(np.float32) / 255.0
            Y = np.array(data_dict[b'labels'])
            return X, Y

    def _load_cifar10(self):
        """
        加载整个 CIFAR-10 数据集（包括训练集和测试集）
        
        返回:
        - X_train: 训练集图像数据
        - Y_train: 训练集标签
        - X_test: 测试集图像数据
        - Y_test: 测试集标签
        """
        X_list, Y_list = [], []
        for i in range(1, 6):
            batch_file = os.path.join(self.data_dir, f'data_batch_{i}')
            X, Y = self._load_cifar_batch(batch_file)
            X_list.append(X)
            Y_list.append(Y)
        X_train = np.concatenate(X_list)
        Y_train = np.concatenate(Y_list)
        X_test, Y_test = self._load_cifar_batch(os.path.join(self.data_dir, 'test_batch'))
        return X_train, Y_train, X_test, Y_test

    def _split_train_val(self, X, Y):
        """
        将训练集划分为训练集和验证集

        返回:
        - X_train: 训练集图像
        - Y_train: 训练集标签
        - X_val: 验证集图像
        - Y_val: 验证集标签
        """
        np.random.seed(self.seed)
        num_samples = X.shape[0]
        indices = np.random.permutation(num_samples)
        split_idx = int(num_samples * (1 - self.val_ratio))
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]
        return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]

    def _load_and_split(self):
        """
        加载数据并划分训练集、验证集和测试集
        
        返回:
        - X_train, Y_train: 训练集
        - X_val, Y_val: 验证集
        - X_test, Y_test: 测试集
        """
        X_train_all, Y_train_all, X_test, Y_test = self._load_cifar10()
        X_train, Y_train, X_val, Y_val = self._split_train_val(X_train_all, Y_train_all)
        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    def get_batches(self, X, Y, shuffle=True):
        """
        返回一个生成器，用于按 batch 返回数据

        参数:
        - X, Y: 数据和标签
        - shuffle: 是否在每轮开始前打乱

        生成:
        - (X_batch, Y_batch): 每次返回一个 batch 的数据
        """
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        if shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(indices)
        for start_idx in range(0, num_samples, self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_idx = indices[start_idx:end_idx]
            yield X[batch_idx], Y[batch_idx]

    def get_train_set(self):
        """返回训练集数据和标签"""
        return self.X_train, self.Y_train

    def get_val_set(self):
        """返回验证集数据和标签"""
        return self.X_val, self.Y_val

    def get_test_set(self):
        """返回测试集数据和标签"""
        return self.X_test, self.Y_test

    def get_label_name(self, label_id):
        """根据标签编号返回对应的类别名"""
        return self.label_names[label_id]
