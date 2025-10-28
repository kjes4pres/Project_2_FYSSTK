import numpy as np


def mse(predict, target):
    return np.mean((predict - target) ** 2)

def mse_der(predict, target):
    assert predict.shape == target.reshape(-1,1).shape, "Not same"
    return 2 * (predict - target) / target.size

def cross_entropy(predict, target):
    return -np.sum(target*np.log(predict))

def cross_entropy_der(predict, target):
    return -target/predict