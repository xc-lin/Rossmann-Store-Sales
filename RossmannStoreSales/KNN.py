import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor


def knn(x_train, y_train):
    reg = KNeighborsRegressor(weights="uniform")
    score = cross_val_score(reg, x_train, y_train, cv=StratifiedKFold(10))
    # reg.fit(x_train, y_train)
    print("10-folder cross validation score: ", score)
    print("mean score: ", np.mean(score))
    # y_hat = reg.predict(x_valid)
    # print(reg.score(x_train, y_train))
    # error = LossFuction.basicRmspe(y_valid, y_hat)
    # print(error)
