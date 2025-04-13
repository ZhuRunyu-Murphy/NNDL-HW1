import numpy as np

class Activation:
    def __init__(self, name):
        self.name = name

    def forward(self, x):
        if self.name == 'relu':
            self.mask = (x > 0)
            return np.maximum(0, x)
        elif self.name == 'sigmoid':
            self.out = 1 / (1 + np.exp(-x))
            return self.out
        elif self.name == 'tanh':
            self.out = np.tanh(x)
            return self.out
        else:
            raise ValueError("Unsupported activation")

    def backward(self, dout):
        if self.name == 'relu':
            return dout * self.mask
        elif self.name == 'sigmoid':
            return dout * self.out * (1 - self.out)
        elif self.name == 'tanh':
            return dout * (1 - self.out ** 2)


class Linear:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * 0.01
        self.b = np.zeros(out_dim)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        dx = dout @ self.W.T
        return dx

    def update(self, lr, l2_lambda):
        self.W -= lr * (self.dW + l2_lambda * self.W)
        self.b -= lr * self.db


class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim, activations):
        """
        input_dim: 输入特征维度
        hidden_dims: list, 两个隐藏层的维度，例如 [256, 128]
        output_dim: 类别数，例如 10
        activations: list, 两个激活函数名称，例如 ['relu', 'relu']
        """
        assert len(hidden_dims) == 2 and len(activations) == 2
        self.layers = []

        dims = [input_dim] + hidden_dims + [output_dim]

        # 第一层 Linear + Activation
        self.layers.append(Linear(dims[0], dims[1]))
        self.layers.append(Activation(activations[0]))

        # 第二层 Linear + Activation
        self.layers.append(Linear(dims[1], dims[2]))
        self.layers.append(Activation(activations[1]))

        # 第三层 Linear (无激活)
        self.layers.append(Linear(dims[2], dims[3]))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x  # 输出 logits

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def update(self, lr, l2_lambda=0.0):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.update(lr, l2_lambda)

    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=1)
