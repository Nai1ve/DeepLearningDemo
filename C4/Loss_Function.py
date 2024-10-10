import numpy as np



def mean_squared_error(y_true, y_pred):
    return 0.5 * np.sum((y_true - y_pred) ** 2)


def cross_entropy(y_true, y_pred):
    delta = 1e-7
    return -np.sum(y_pred * np.log(y_true+delta))

t = [0,0,1,0,0,0,0,0,0,0]
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
print(cross_entropy(np.array(y), np.array(t)))

y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
print(cross_entropy(np.array(y), np.array(t)))