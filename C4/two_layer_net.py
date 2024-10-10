import sys,os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size,weight_init_std = 0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['b2'] = np.zeros(output_size)
        #print(f"W1:{self.params['W1']},W1 dim :{np.ndim(self.params['W1'])}")
        #print(f"W2:{self.params['W2']},W2 dim :{np.ndim(self.params['W2'])}")
        #print(f"b1:{self.params['b1']},b1 dim :{np.ndim(self.params['b1'])}")
        #print(f"b2:{self.params['b2']},b2 dim :{np.ndim(self.params['b2'])}")


    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # x:输入，t：监督
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y,t)


    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    #x:输入数据 t：监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

# net = TwoLayerNet(784,100,10)
# x = np.random.rand(100,784)
# print(f"x:{x},x dim :{np.ndim(x)}")
# y =net.predict(x)
# print(f"y:{y},y dim :{np.ndim(y)}")
#
# t = np.random.rand(100,10)
# print(f"t:{t},t dim :{np.ndim(t)}")
# grads = net.numerical_gradient(x, t)
# print(f"grads:{grads},grads dim :{np.ndim(grads)}")