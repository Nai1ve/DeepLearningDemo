import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import pickle
import numpy as np
from C3.ThreeLayeredNeuralNetworkDemo import sigmoid,softmax

# (训练图像，训练标签)，（测试图像，测试标签）
(x_train,t_train) , (x_test,t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

def get_data():
    (x_train,t_train), (x_test,t_test) = load_mnist(normalize=True,flatten=True, one_hot_label=False)
    return x_test,t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)

    return y

x,t = get_data()
network = init_network()

accuracy_count = 0

for i in range(len(x)):
    y = predict(network,x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_count += 1

print("Accuracy:"+str(float(accuracy_count) / len(x)))
