import numpy as np


def mse(predict, target):
    return np.mean((predict - target) ** 2)

def mse_der(predict, target):
    assert predict.shape == target.reshape(-1,1).shape, "Not same"
    return 2 * (predict - target) / target.size