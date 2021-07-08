import numpy as np
import pandas

from sklearn import linear_model
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from RossmannStoreSales import LossFuction


def preprocess(x_train, y_train):
    one_hot_code_features = ["DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday", "StoreType", "Assortment",
                             "Promo2", "IsInPromo", "Year", "Month", "Day", "Open", "Promo2SinceWeek",
                             "Promo2SinceYear"]
    one_hot_part = pandas.get_dummies(x_train, columns=one_hot_code_features)

    mm = MinMaxScaler()
    # data_CompetitionDistance = train_data[["CompetitionDistance"]]
    scalered_dis = mm.fit_transform(x_train[["CompetitionDistance"]])
    scalered_dis = pandas.DataFrame(scalered_dis, columns=["CompetitionDistance"])

    # XgboostModel.xgboostModel(x_train, y_train, x_valid, y_valid)
    x_train = pandas.concat([one_hot_part, scalered_dis], axis=1)
    return x_train, y_train


def linearRegression(x_train, y_train):
    x_train, y_train = preprocess(x_train, y_train)
    reg = linear_model.LinearRegression()
    score = cross_val_score(reg, x_train, y_train, cv=StratifiedKFold(10))
    # reg.fit(x_train, y_train)
    print("10-folder cross validation score: ", score)
    print("mean score: ", np.mean(score))
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
