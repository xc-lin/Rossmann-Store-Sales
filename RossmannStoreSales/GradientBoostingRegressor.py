import numpy as np
import pandas
import sklearn
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor

from RossmannStoreSales import LossFuction


def preprocess(x_train, y_train):
    mm = MinMaxScaler()
    # data_CompetitionDistance = train_data[["CompetitionDistance"]]
    scalered_dis = mm.fit_transform(x_train[["CompetitionDistance"]])
    x_train["CompetitionDistance"] = pandas.DataFrame(scalered_dis, columns=["CompetitionDistance"])
    return x_train, y_train


def GradientBoosting(x_train, y_train):
    # x_train, y_train = preprocess(x_train, y_train)
    reg = GradientBoostingRegressor()
    score = cross_val_score(reg, x_train, y_train, cv=StratifiedKFold(10))
    # reg.fit(x_train, y_train)
    print("10-folder cross validation score: ", score)
    print("mean score: ", np.mean(score))
