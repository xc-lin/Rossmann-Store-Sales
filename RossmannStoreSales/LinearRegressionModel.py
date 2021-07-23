import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, StratifiedKFold

from RossmannStoreSales.Compare import compareResult
from RossmannStoreSales.Preprocess import preprocess


def linearRegression(x_train_v, y_train_v, x_valid, y_valid, nfolds):
    x_train_v = preprocess(x_train_v)
    reg = linear_model.LinearRegression()
    score = cross_val_score(reg, x_train_v, y_train_v, cv=StratifiedKFold(nfolds))
    x_valid, y_valid = preprocess(x_valid, y_valid)
    y_hat = reg.predict(x_valid)
    plt.title("linearRegression")
    plt.plot(y_hat[::200])
    plt.plot(y_valid.iloc[::200].values)
    plt.legend(["y_hat (linearRegression)", "real"])
    plt.show()
    print("linearRegression:")
    print("10-folder cross validation score: ", score)
    print("mean score: ", np.mean(score))


# def generateLinearRegression(x_train, y_train, x_valid, y_valid):
#     x_train, y_train = preprocess(x_train, y_train)
#     reg = linear_model.LinearRegression()
#     reg.fit(x_train, y_train)
#     compareResult(reg, x_valid, y_valid, "LinearRegression")
#     joblib.dump(reg, '../model/LinearRegression.pkl')


# def sgdRegression(x_train, y_train):
#     x_train, y_train = preprocess(x_train, y_train)
#     reg = linear_model.SGDRegressor()
#     score = cross_val_score(reg, x_train, y_train, cv=StratifiedKFold(2))
#     print("LogisticRegression:")
#     print("10-folder cross validation score: ", score)
#     print("mean score: ", np.mean(score))
#
#
# def generateSGDRegression(x_train, y_train, x_valid, y_valid):
#     x_train, y_train = preprocess(x_train, y_train)
#     reg = linear_model.SGDRegressor()
#     reg.fit(x_train, y_train)
#     compareResult(reg, x_valid, y_valid, "SGDRegressor")
#     joblib.dump(reg, '../model/SGDRegressor.pkl')


def ridgeRegression(x_train, y_train, alpha):
    x_train = preprocess(x_train)
    reg = linear_model.Ridge(alpha=alpha)
    score = cross_val_score(reg, x_train, y_train, cv=StratifiedKFold(10))
    # reg.fit(x_train, y_train)
    print("ridgeRegression:")
    print("10-folder cross validation score: ", score)
    mean_score = np.mean(score)
    print("mean score: ", mean_score)
    return mean_score


def generateRidgeRegression(x_train, y_train, alpha, x_valid, y_valid):
    x_train, y_train = preprocess(x_train, y_train)
    reg = linear_model.Ridge(alpha=alpha)
    reg.fit(x_train, y_train)
    compareResult(reg, x_valid, y_valid, "RidgeRegression")
    joblib.dump(reg, '../model/RidgeRegression.pkl')


def lassoRegression(x_train, y_train, alpha):
    x_train, y_train = preprocess(x_train, y_train)
    reg = linear_model.Lasso(alpha=alpha)
    score = cross_val_score(reg, x_train, y_train, cv=StratifiedKFold(10))
    # reg.fit(x_train, y_train)
    print("lassoRegression:")
    print("10-folder cross validation score: ", score)
    mean_score = np.mean(score)
    print("mean score: ", mean_score)
    return mean_score


def generateLassoRegression(x_train, y_train, alpha, x_valid, y_valid):
    x_train, y_train = preprocess(x_train, y_train)
    reg = linear_model.Lasso(alpha=alpha)
    reg.fit(x_train, y_train)
    compareResult(reg, x_valid, y_valid, "LassoRegression")
    joblib.dump(reg, '../model/LassoRegression.pkl')

#
# def linearRegressionPerStore(train, valid):
#     rossmann_dic = dict(list(train.groupby('Store')))
#     valid_dic = dict(list(valid.groupby('Store')))
#     ss = StandardScaler()
#     errors = []
#     for i in rossmann_dic:
#         store = rossmann_dic[i]
#         valid_store = valid_dic[i]
#         # define training and testing sets
#         x_train = store.drop(["Sales", "Store"], axis=1)
#
#         # x_train = ss.fit_transform(x_train)
#         y_train = store["Sales"]
#         x_valid = valid_store.drop(["Sales", "Store"], axis=1)
#         # x_valid = ss.fit_transform(x_valid)
#         y_valid = valid_store["Sales"]
#
#         # Linear Regression
#         reg = linear_model.LinearRegression()
#         reg.fit(x_train, y_train)
#         y_hat = reg.predict(x_valid)
#         error = LossFuction.basicRmspe(y_valid, y_hat)
#         # print(y_hat)
#         # print(y_valid)
#         # print()
#         errors.append(error)
#         # print(error)
#     print(np.mean(errors))
#
#
# def ridgeRegressionPerStore(train, valid, alpha):
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
#         reg = linear_model.Ridge(alpha=alpha)
#         reg.fit(x_train, y_train)
#         y_hat = reg.predict(x_valid)
#         error = LossFuction.basicRmspe(y_valid, y_hat)
#         # print(y_hat)
#         # print(y_valid)
#         # print()
#         errors.append(error)
#         # print(error)
#     print(np.mean(errors))
