from sklearn import linear_model

from RossmannStoreSales import LossFuction


def linearRegression(x_train, y_train, x_valid, y_valid):
    reg = linear_model.LinearRegression()

    reg.fit(x_train, y_train)
    y_hat = reg.predict(x_valid)
    print(reg.score(x_train, y_train))
    error = LossFuction.basicRmspe(y_valid, y_hat)
    print(error)


def ridgeRegression(x_train, y_train, x_valid, y_valid,alpha):
    reg = linear_model.Ridge(alpha=alpha)

    reg.fit(x_train, y_train)
    y_hat = reg.predict(x_valid)

    error = LossFuction.basicRmspe(y_valid, y_hat)
    print(error)
