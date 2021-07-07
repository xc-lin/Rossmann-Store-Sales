import numpy as np


def basicRmspe(y, y_hat):
    result = (np.mean(((y - y_hat) / y) ** 2)) ** 0.5
    return result


def rmspe(y_hat, y):
    y = np.expm1(y.get_label())
    y_hat = np.expm1(y_hat)
    return "rmspe", basicRmspe(y, y_hat)