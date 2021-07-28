import numpy as np


def handleZero(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1. / (y[ind])
    return w


def basicRmspe(y, y_hat):
    one_over_y = handleZero(y)
    s1 = one_over_y * (y - y_hat)
    s2 = np.mean(s1 ** 2)
    result = np.sqrt(s2)
    return result


def rmspe(y_hat, y):
    y = y.get_label()
    y = np.exp(y) - 1
    y_hat = np.exp(y_hat) - 1
    result = basicRmspe(y, y_hat)
    return "rmspe", result
