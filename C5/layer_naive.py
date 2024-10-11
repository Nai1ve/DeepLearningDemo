import numpy as np
from common.functions import softmax, cross_entropy_error

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None


    def forward(self, x,y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass


    def forward(self, x,y):
        out = x + y
        return out


    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


# 激活函数层
class Relu:
    def __init__(self):
        self.mask = None


    def forward(self, x):
        """
        实现Relu激活函数
        :param x:  Numpy数组
        :return:
        """
        # 创建一个与x形状相同的mask，并将所有小于等于0的元素置为True，其余元素置为False
        self.mask = (x <= 0)
        out = x.copy()
        # 将副本out中mask为True的元素置为0 Numpy中的布尔索引
        out[self.mask] = 0

        return out


    def backword(self, dout):
        """
        实现Relu的反向传播
        :param dout:  损失函数对Relu的导数 Numpy数组
        :return:
        """
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        """
        实现Sigmoid激活函数
        :param x:  Numpy数组
        :return:
        """
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        """
        实现Sigmoid的反向传播
        :param dout:  损失函数对Sigmoid的导数 Numpy数组
        :return:
        """
        # 化简整理后的公式
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None


    def forward(self, x):
        """
        实现Affine层的前向传播
        :param x:  Numpy数组
        :return:
        """
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out


    def backward(self, dout):
        """
        实现Affine层的反向传播
        :param dout:  损失函数对Affine层的导数 Numpy数组
        :return:
        """
        dx = np.dot(dout, self.W.T)
        # self.x.T表示x的转置
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 损失
        self.y = None     # softmax的输出
        self.t = None     # 标签


    def forward(self, x, t):
        """
        实现SoftmaxWithLoss层的前向传播
        :param x:  Softmax层的输出 Numpy数组
        :param t:  标签 Numpy数组
        :return:
        """
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss


    def backward(self, dout=1):
        """
        实现SoftmaxWithLoss层的反向传播
        :param dout:  损失函数对SoftmaxWithLoss层的导数，默认为1
        :return:
        """
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx