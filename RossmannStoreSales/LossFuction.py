import numpy as np


def handleZero(y):
    one_over_y = []
    for i in y:
        if i != 0:
            one_over_y.append(1 / i)
        else:
            one_over_y.append(0)
    return one_over_y


def basicRmspe(y, y_hat):
    one_over_y = handleZero(y)
    s1 = (y - y_hat) * one_over_y
    s2 = s1 ** 2
    s3 = np.mean(s2)
    result = s3 ** 0.5
    return result


# def rmspe(y_hat, y):
#     y = np.expm1(y.get_label())
#     y_hat = np.expm1(y_hat)
#     return "rmspe", basicRmspe(y, y_hat)
