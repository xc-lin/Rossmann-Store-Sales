import math

import numpy as np


def basicRmspe(y, y_hat):
    result = math.sqrt(np.mean(((y - y_hat) / y) ** 2))
    return result


def rmspe(y, y_hat):
    y = np.log1p(y.get_label())
    y_hat = np.log1p(y_hat)
    return "rmspe", basicRmspe(y, y_hat)
