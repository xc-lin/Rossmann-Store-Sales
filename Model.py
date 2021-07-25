import time

import joblib
import numpy as np
import pandas
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from xgboost import plot_importance

import LossFuction


# warnings.filterwarnings("ignore")


def preprocess(x_train):
    one_hot_code_features = ["DayOfWeek", "StateHoliday", "StoreType", "Assortment",
                             "Year", "Month", "Day", "Promo2SinceWeek",
                             "Promo2SinceYear"]
    x_train = pandas.get_dummies(x_train, columns=one_hot_code_features)
    x_train["CompetitionDistance"] = (x_train["CompetitionDistance"] - x_train["CompetitionDistance"].min()) / (
            x_train["CompetitionDistance"].max() - x_train["CompetitionDistance"].min())
    return x_train


def preprocessNormalize(x_train):
    x_train["CompetitionDistance"] = (x_train["CompetitionDistance"] - x_train["CompetitionDistance"].min()) / (
            x_train["CompetitionDistance"].max() - x_train["CompetitionDistance"].min())

    return x_train


def linearRegression(x_train_v, y_train_v, x_valid, y_valid, nfolds):
    print("Start normalizing and one-hot code...")
    x_train_v = preprocess(x_train_v)
    x_valid = preprocess(x_valid)
    print("normalization and one-hot code is finished")
    print()
    print("Start fitting and cross validation...")
    reg = linear_model.LinearRegression()
    score = cross_val_score(reg, x_train_v, y_train_v, cv=StratifiedKFold(nfolds))
    print(reg.__class__.__name__)
    print()
    print("*" * 5, "%d-folder cross validation score: " % nfolds, score, "*" * 5)
    print("*" * 5, "mean score: ", np.mean(score), "*" * 5)
    print("*" * 5, "fit and cross validation is finished", "*" * 5)
    print()
    predictDataPlot(reg, x_train_v, y_train_v, x_valid, y_valid)


def decisionTree(x_train_v, y_train_v, x_valid, y_valid, nfolds):
    print("Start normalize data...")
    x_train_v = preprocessNormalize(x_train_v)
    x_valid = preprocessNormalize(x_valid)
    print("data normalization is finished")
    print()
    print("Start fitting and cross validation...")
    reg = DecisionTreeRegressor()
    score = cross_val_score(reg, x_train_v, y_train_v, cv=StratifiedKFold(nfolds))
    print(reg.__class__.__name__)
    print()
    print("*" * 5, "%d-folder cross validation score: " % nfolds, score, "*" * 5)
    print("*" * 5, "mean score: ", np.mean(score), "*" * 5)
    print("*" * 5, "fit and cross validation is finished", "*" * 5)
    print()

    predictDataPlot(reg, x_train_v, y_train_v, x_valid, y_valid)


def extraTrees(x_train_v, y_train_v, x_valid, y_valid, nfolds):
    print("Start normalize data...")
    x_train_v = preprocessNormalize(x_train_v)
    x_valid = preprocessNormalize(x_valid)
    print("data normalization is finished")
    print()
    print("Start fitting and cross validation...")
    reg = ExtraTreesRegressor()
    t5 = time.time()
    score = cross_val_score(reg, x_train_v, y_train_v, cv=StratifiedKFold(nfolds))
    t6 = time.time()
    print(reg.__class__.__name__)
    print()
    print("*" * 5, "%d-folder cross validation score: " % nfolds, score, "*" * 5)
    print("*" * 5, "mean score: ", np.mean(score), "*" * 5)
    print("*" * 5, "fit and cross validation is finished", "*" * 5)
    print()
    predictDataPlot(reg, x_train_v, y_train_v, x_valid, y_valid)


def gradientBoosting(x_train_v, y_train_v, x_valid, y_valid, nfolds):
    print("Start normalize data...")
    x_train_v = preprocessNormalize(x_train_v)
    x_valid = preprocessNormalize(x_valid)
    print("data normalization is finished")
    print()
    print("Start fitting and cross validation...")
    reg = GradientBoostingRegressor()
    t5 = time.time()
    score = cross_val_score(reg, x_train_v, y_train_v, cv=StratifiedKFold(nfolds))
    t6 = time.time()
    print(reg.__class__.__name__)
    print()
    print("*" * 5, "%d-folder cross validation score: " % nfolds, score, "*" * 5)
    print("*" * 5, "mean score: ", np.mean(score), "*" * 5)
    print("*" * 5, "fit and cross validation is finished", "*" * 5)
    print()
    predictDataPlot(reg, x_train_v, y_train_v, x_valid, y_valid)


def randomForest(x_train_v, y_train_v, x_valid, y_valid, nfolds):
    print("Start normalize data...")
    x_train_v = preprocess(x_train_v)
    x_valid = preprocess(x_valid)
    print("data normalization is finished")
    print()
    print("Start fitting and cross validation...")
    reg = RandomForestRegressor(n_jobs=-1)
    t5 = time.time()
    score = cross_val_score(reg, x_train_v, y_train_v, cv=StratifiedKFold(nfolds))
    t6 = time.time()
    print(reg.__class__.__name__)
    print()
    print("*" * 5, "%d-folder cross validation score: " % nfolds, score, "*" * 5)
    print("*" * 5, "mean score: ", np.mean(score), "*" * 5)
    print("*" * 5, "fit and cross validation is finished", "*" * 5)
    print()
    predictDataPlot(reg, x_train_v, y_train_v, x_valid, y_valid)


def xgboost(x_train_v, y_train_v, x_valid, y_valid, test_data):
    print("Start training xgboost model...")
    # y_train_v = np.log(1 + y_train_v)
    # y_valid = np.log(1 + y_valid)
    train_matrix = xgb.DMatrix(x_train_v, y_train_v)
    valid_matrix = xgb.DMatrix(x_valid, y_valid)

    param = {'max_depth': 9,
             'eta': 0.06,
             'subsample': 0.75,
             'colsample_bytree': 0.6,
             'objective': 'reg:squarederror', }
    params = list(param.items())
    num_boost_round = 10000
    evals = [(train_matrix, 'train'), (valid_matrix, 'valid')]
    reg = xgb.train(params, train_matrix, num_boost_round, evals,
                    feval=LossFuction.rmspe, verbose_eval=1, early_stopping_rounds=50)
    # joblib.dump(reg, 'Xgboost.pkl')

    print("xgboost training is finished")
    print()

    print("Start generating plots ...")
    # reg = joblib.load("../model/Xgboost.pkl")

    y_valid_hat = reg.predict(valid_matrix)
    correction_factor = 0.975
    plt.title("Xgboost predict data after correction")
    plt.plot((np.exp(y_valid_hat[::900]) - 1) * correction_factor)
    plt.plot(np.exp(y_valid.values[::900]) - 1)
    plt.legend(["y_hat(xgboost)", "real"])
    plt.show()
    # plot Feature Importance
    a, ax = plt.subplots(1, 1, figsize=(20, 8))
    plot_importance(reg, ax=ax)
    plt.show()
    print("plots generation is finished...")
    print()

    print("Start generating submission.csv ...")
    submission_df = test_data['Id'].reset_index()
    submission_df['Id'] = submission_df['Id'].astype('int')

    test_matrix = xgb.DMatrix(test_data.drop("Id", axis=1))
    y_test_hat = reg.predict(test_matrix)

    submission_df['Sales'] = (np.exp(y_test_hat) - 1) * correction_factor
    submission_df.sort_values('Id', inplace=True)
    submission_df[['Id', 'Sales']].to_csv('submission2.csv', index=False)
    print("submission.csv generation is finished...")
    print()


def xgboostPredict(x_valid, y_valid, test_data):
    print("Start generating plots ...")
    # y_valid = np.log(1 + y_valid)
    valid_matrix = xgb.DMatrix(x_valid, y_valid)

    reg = joblib.load("Xgboost.pkl")
    y_valid_hat = reg.predict(valid_matrix)
    correction_factor = 0.975
    plt.title("Xgboost predict data after correction")
    plt.plot((np.exp(y_valid_hat[::900]) - 1) * correction_factor)
    plt.plot(np.exp(y_valid.values[::900]) - 1)
    plt.legend(["y_hat(xgboost)", "real"])
    plt.show()
    # plot Feature Importance
    a, ax = plt.subplots(1, 1, figsize=(20, 8))
    plot_importance(reg, ax=ax)
    plt.show()
    print("plots generation is finished...")
    print()

    print("Start generating submission.csv ...")
    submission_df = test_data['Id'].reset_index()
    submission_df['Id'] = submission_df['Id'].astype('int')

    test_matrix = xgb.DMatrix(test_data.drop("Id", axis=1))
    y_test_hat = reg.predict(test_matrix)

    submission_df['Sales'] = (np.exp(y_test_hat) - 1) * correction_factor
    submission_df.sort_values('Id', inplace=True)
    submission_df[['Id', 'Sales']].to_csv('submission2.csv', index=False)
    print("submission.csv generation is finished...")
    print()


def predictDataPlot(reg, x_train_v, y_train_v, x_valid, y_valid):
    print("Start generating the plot of predict and true data...")
    reg.fit(x_train_v, y_train_v)
    y_hat = reg.predict(x_valid)
    plt.title(reg.__class__.__name__)
    plt.plot(y_hat[::900])
    plt.plot(y_valid.iloc[::900].values)
    plt.legend(["y_hat (%s)" % reg.__class__.__name__, "real"])
    plt.show()
    print("plot generation is finished...")
    print()
