import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt


# 加载数据集
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []
train_size = x_train.shape[0]
batch_size = 100
# 每个epoch的重复次数
iter_per_epoch = max(train_size / batch_size, 1)

# 设置超参数
iters_num = 10000
learning_rate = 0.1

# 创建模型
net = TwoLayerNet(input_size=28*28, hidden_size=50, output_size=10)

# 开始训练
for i in range(iters_num):
    # 随机选择batch_size个样本
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    # grad = net.numerical_gradient(x_batch, t_batch)
    # 反向传播版本
    grad = net.gradient(x_batch, t_batch)

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        net.params[key] -= learning_rate * grad[key]

    # 记录训练损失
    loss = net.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 打印训练信息
    if i % 1000 == 0:
        print("train loss:" + str(loss))

    if i % iter_per_epoch == 0:
        # 计算训练集准确率
        train_acc = net.accuracy(x_train, t_train)
        test_acc = net.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc:" + str(train_acc) + " test acc:" + str(test_acc))


# 绘制图形
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

plt.plot(train_loss_list)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.show()