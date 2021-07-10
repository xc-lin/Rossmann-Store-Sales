from matplotlib import pyplot as plt

from RossmannStoreSales.Preprocess import preprocess, preprocessMM


def compareResult(reg, x_valid, y_valid, model):
    x_valid, y_valid = preprocess(x_valid, y_valid)
    y_hat = reg.predict(x_valid)
    plt.title(model)
    plt.plot(y_hat[::100])
    plt.plot(y_valid.iloc[::100].values)
    plt.legend(["y_hat({})".format(model), "real"])
    plt.show()


def compareResultMM(reg, x_valid, y_valid, model):
    x_valid, y_valid = preprocessMM(x_valid, y_valid)
    y_hat = reg.predict(x_valid)
    plt.title(model)
    plt.plot(y_hat[::100])
    plt.plot(y_valid.iloc[::100].values)
    plt.legend(["y_hat({})".format(model), "real"])
    plt.show()


def compareResultPure(reg, x_valid, y_valid, model):
    y_hat = reg.predict(x_valid)
    plt.title(model)
    plt.plot(y_hat[::100])
    plt.plot(y_valid.iloc[::100].values)
    plt.legend(["y_hat({})".format(model), "real"])
    plt.show()
