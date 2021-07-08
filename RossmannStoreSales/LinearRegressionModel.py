import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from RossmannStoreSales import LossFuction


def linearRegression(x_train, y_train, x_valid, y_valid):
    reg = linear_model.LinearRegression()
    score = cross_val_score(reg, x_train, y_train, cv=StratifiedKFold(5))
    reg.fit(x_train, y_train)
    print(score)
    # y_hat = reg.predict(x_valid)
    # print(reg.score(x_train, y_train))
    # error = LossFuction.basicRmspe(y_valid, y_hat)
    # print(error)


def ridgeRegression(x_train, y_train, x_valid, y_valid, alpha):
    reg = linear_model.Ridge(alpha=alpha)

    reg.fit(x_train, y_train)
    y_hat = reg.predict(x_valid)

    error = LossFuction.basicRmspe(y_valid, y_hat)
    print(error)


def linearRegressionPerStore(train, valid):
    rossmann_dic = dict(list(train.groupby('Store')))
    valid_dic = dict(list(valid.groupby('Store')))
    ss = StandardScaler()
    errors = []
    for i in rossmann_dic:
        store = rossmann_dic[i]
        valid_store = valid_dic[i]
        # define training and testing sets
        x_train = store.drop(["Sales", "Store"], axis=1)

        # x_train = ss.fit_transform(x_train)
        y_train = store["Sales"]
        x_valid = valid_store.drop(["Sales", "Store"], axis=1)
        # x_valid = ss.fit_transform(x_valid)
        y_valid = valid_store["Sales"]

        # Linear Regression
        reg = linear_model.LinearRegression()
        reg.fit(x_train, y_train)
        y_hat = reg.predict(x_valid)
        error = LossFuction.basicRmspe(y_valid, y_hat)
        # print(y_hat)
        # print(y_valid)
        # print()
        errors.append(error)
        # print(error)
    print(np.mean(errors))


def ridgeRegressionPerStore(train, valid, alpha):
    rossmann_dic = dict(list(train.groupby('Store')))
    valid_dic = dict(list(valid.groupby('Store')))
    errors = []
    for i in rossmann_dic:
        store = rossmann_dic[i]
        valid_store = valid_dic[i]
        # define training and testing sets
        x_train = store.drop(["Sales", "Store"], axis=1)
        y_train = store["Sales"]
        x_valid = valid_store.drop(["Sales", "Store"], axis=1)
        y_valid = valid_store["Sales"]

        # Linear Regression
        reg = linear_model.Ridge(alpha=alpha)
        reg.fit(x_train, y_train)
        y_hat = reg.predict(x_valid)
        error = LossFuction.basicRmspe(y_valid, y_hat)
        # print(y_hat)
        # print(y_valid)
        # print()
        errors.append(error)
        # print(error)
    print(np.mean(errors))
