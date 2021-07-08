import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from RossmannStoreSales import LossFuction


def decisionTree(x_train, y_train, x_valid, y_valid):
    DT = DecisionTreeClassifier()
    DT.fit(x_train, y_train)
    y_hat=DT.predict(x_valid)
    error = LossFuction.basicRmspe(y_valid, y_hat)
    print(error)


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


def randomForest(x_train, y_train, x_valid, y_valid):
    DT = DecisionTreeClassifier()
    DT.fit(x_train, y_train)
    y_hat=DT.predict(x_valid)
    error = LossFuction.basicRmspe(y_valid, y_hat)
    print(error)


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
