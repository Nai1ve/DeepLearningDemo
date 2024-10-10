import numpy as np


# 不好的实现示例
def numerical_diff_bad(f, x):
    h = 10e-50 # 导致计算机的舍入误差 10^-4
    return (f(x+h) - f(x))/h # 存在误差，我们计算导数时会用lim h->0，但现实中没有，解决办法为我们扩大h

# 好的实现
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x - h)) / (2 *h)


def function_2(x):
    return x[0]**2 + x[1]**2


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for i in range(x.size):
        tmp_val = x[i]
        x[i] = tmp_val + h
        fxh1 = f(x)

        x[i] = tmp_val - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2)/(2*h)
        x[i] = tmp_val
    return grad


def gradient_descent(f,init_x,lr=0.01,setp_num = 100):
    x = init_x

    for i in range(setp_num):
        grad = numerical_gradient(f,x)
        x -= lr * grad
    return x

init_x = np.array([-3.0,4.0])
print(gradient_descent(function_2,init_x,lr=0.1,setp_num=100))