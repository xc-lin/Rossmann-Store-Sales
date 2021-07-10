import time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold

from RossmannStoreSales import LossFuction
from RossmannStoreSales.Compare import compareResult, compareResultMM, compareResultPure
from RossmannStoreSales.Preprocess import preprocessMM


def randomForest(x_train, y_train):
    x_train, y_train = preprocessMM(x_train, y_train)
    reg = RandomForestRegressor()
    t5 = time.time()
    score = cross_val_score(reg, x_train, y_train, cv=StratifiedKFold(10))
    t6 = time.time()
    print("10-folder cross validation score: ", score)
    print("mean score: ", np.mean(score))
    print("time: ", t6 - t5)


def generateRandomForest(x_train, y_train, x_valid, y_valid):
    x_train, y_train = preprocessMM(x_train, y_train)
    reg = RandomForestRegressor()
    reg.fit(x_train, y_train)
    compareResultMM(reg, x_valid, y_valid, "RandomForestRegressor")
    # joblib.dump(reg, '../model/RandomForestRegressor.pkl')


def extraTrees(x_train, y_train):
    x_train, y_train = preprocessMM(x_train, y_train)
    reg = ExtraTreesRegressor()
    t5 = time.time()
    score = cross_val_score(reg, x_train, y_train, cv=StratifiedKFold(10))
    t6 = time.time()
    print("10-folder cross validation score: ", score)
    print("mean score: ", np.mean(score))
    print("time: ", t6 - t5)


def generateExtraTrees(x_train, y_train, x_valid, y_valid):
    x_train, y_train = preprocessMM(x_train, y_train)
    reg = ExtraTreesRegressor()
    reg.fit(x_train, y_train)
    compareResultMM(reg, x_valid, y_valid, "ExtraTreesRegressor")
    joblib.dump(reg, '../model/ExtraTreesRegressor.pkl')


def gradientBoosting(x_train, y_train):
    # x_train, y_train = preprocessMM(x_train, y_train)
    reg = GradientBoostingRegressor()
    score = cross_val_score(reg, x_train, y_train, cv=StratifiedKFold(10))
    # reg.fit(x_train, y_train)
    print("10-folder cross validation score: ", score)
    print("mean score: ", np.mean(score))


def generateGradientBoosting(x_train, y_train, x_valid, y_valid):
    reg = GradientBoostingRegressor()
    reg.fit(x_train, y_train)
    compareResultPure(reg, x_valid, y_valid, "GradientBoostingRegressor")
    joblib.dump(reg, '../model/GradientBoostingRegressor.pkl')












#
# def randomForestPerStore(train, valid):
#     rossmann_dic = dict(list(train.groupby('Store')))
#     valid_dic = dict(list(valid.groupby('Store')))
#     errors = []
#     for i in rossmann_dic:
#         store = rossmann_dic[i]
#         valid_store = valid_dic[i]
#         # define training and testing sets
#         x_train = store.drop(["Sales", "Store"], axis=1)
#         y_train = store["Sales"]
#         x_valid = valid_store.drop(["Sales", "Store"], axis=1)
#         y_valid = valid_store["Sales"]
#
#         # Linear Regression
#         reg = RandomForestRegressor()
#         reg.fit(x_train, y_train)
#         y_hat = reg.predict(x_valid)
#         error = LossFuction.basicRmspe(y_valid, y_hat)
#         # print(y_hat)
#         # print(y_valid)
#         # print()
#         errors.append(error)
#         # print(error)
#     print(np.mean(errors))
