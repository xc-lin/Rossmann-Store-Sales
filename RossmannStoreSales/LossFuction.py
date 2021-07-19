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


def rmspe(y_hat, y):
    y = y.get_label()
    y = np.exp(y) - 1
    y_hat = np.exp(y_hat) - 1
    return "rmspe", basicRmspe(y, y_hat)

#
# Thanks for providing this in the forum
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1. / (y[ind] ** 2)
    return w


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return rmspe


def rmspe_xg(yhat, y):
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return "rmspe", rmspe


