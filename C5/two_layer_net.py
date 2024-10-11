import sys,os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        初始化两层神经网络
        :param input_size:      输入层大小
        :param hidden_size:     隐藏层大小
        :param output_size:     输出层大小
        :param weight_init_std: 权重的标准差
        """

        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        # 记录中间变量
        self.last_layer = SoftmaxWithLoss()


    def predict(self, x):
        """
        预测
        :param x: 输入数据
        :return: 输出数据
        """
        for layer in self.layers.values():
            x = layer.forward(x)

        return x


    def loss(self, x, t):
        """
        计算损失
        :param x: 输入数据
        :param t: 监督数据
        :return: 损失值
        """
        y = self.predict(x)

        return self.last_layer.forward(y, t)


    def accuracy(self, x, t):
        """
        计算准确度
        :param x: 输入数据
        :param t: 监督数据
        :return: 准确度
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


    def numerical_gradient(self, x, t):
        """
        计算梯度
        :param x: 输入数据
        :param t: 监督数据
        :return: 权重的梯度
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads


    def gradient(self, x, t):
        """
        计算梯度
        :param x: 输入数据
        :param t: 监督数据
        :return: 权重的梯度
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads