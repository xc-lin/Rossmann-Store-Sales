import time

import numpy as np
import pandas
import sklearn
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from RossmannStoreSales import LossFuction


def preprocess(x_train, y_train):
    mm = MinMaxScaler()
    # data_CompetitionDistance = train_data[["CompetitionDistance"]]
    scalered_dis = mm.fit_transform(x_train[["CompetitionDistance"]])
    x_train["CompetitionDistance"] = pandas.DataFrame(scalered_dis, columns=["CompetitionDistance"])
    return x_train, y_train


def decisionTree(x_train, y_train):
    # x_train, y_train = preprocess(x_train, y_train)
    reg = DecisionTreeRegressor()
    score = cross_val_score(reg, x_train, y_train, cv=StratifiedKFold(10))
    # reg.fit(x_train, y_train)
    print("10-folder cross validation score: ", score)
    print("mean score: ", np.mean(score))


def decisionTreePerStore(train, valid):
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
        DT = DecisionTreeClassifier()
        DT.fit(x_train, y_train)
        y_hat = DT.predict(x_valid)
        error = LossFuction.basicRmspe(y_valid, y_hat)
        # print(y_hat)
        # print(y_valid)
        # print()
        errors.append(error)
        # print(error)
    print(np.mean(errors))


def randomForest(x_train, y_train):
    x_train, y_train = preprocess(x_train, y_train)
    reg = RandomForestRegressor()
    t5 = time.time()
    score = cross_val_score(reg, x_train, y_train, cv=StratifiedKFold(10))
    t6 = time.time()
    print("10-folder cross validation score: ", score)
    print("mean score: ", np.mean(score))
    print("time: ", t6 - t5)


def randomForestPerStore(train, valid):
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
        reg = RandomForestClassifier()
        reg.fit(x_train, y_train)
        y_hat = reg.predict(x_valid)
        error = LossFuction.basicRmspe(y_valid, y_hat)
        # print(y_hat)
        # print(y_valid)
        # print()
        errors.append(error)
        # print(error)
    print(np.mean(errors))
