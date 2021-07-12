import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from RossmannStoreSales import LossFuction
from RossmannStoreSales.Preprocess import preprocess, preprocessMM


def optimization(x_train, y_train, model):
    reg = joblib.load("../model/%s.pkl" % model)
    importance_metrics = pandas.Series(reg.feature_importances_, index=x_train.columns).sort_values()
    plt.figure(figsize=(14, 7))
    # plt.style.use('ggplot')
    importance_metrics[importance_metrics > 0.01].plot.barh()  # 帅选重要程度超过0.005的
    plt.xlabel('importance')
    plt.ylabel('feature')
    plt.title('%s feature importance' % model)
    plt.show()


def optimizationRFTotal(valid):
    reg = joblib.load("../model/RandomForestRegressor.pkl")
    y_valid = valid["Sales"]
    x_valid = valid.drop("Sales", axis=1)
    x_valid, y_valid = preprocessMM(x_valid, y_valid)
    w = 0.93
    ws = []
    errors = []
    while w < 0.95:
        y_hat = reg.predict(x_valid) * w
        ws.append(w)
        error = LossFuction.basicRmspe(y_valid.values, y_hat)
        errors.append(error)
        w += 0.0001
        print("w = %f, error = %f" % (w, error))
    min_error = min(errors)
    w = ws[np.argmin(errors)]
    print("*" * 30)
    print("min:w = %f, error = %f" % (w, min_error))
    plt.title("Model Optimization")
    plt.plot(ws, errors)
    plt.xlabel("W")
    plt.ylabel("Error")
    plt.show()
    y_hat = reg.predict(x_valid)
    plt.plot(y_hat[::200])
    plt.plot(y_hat[::200] * w)
    plt.plot(y_valid.iloc[::200].values)
    plt.legend(["original y_hat(RandomForestRegressor)", "apply overall correction factor", "real"])
    plt.show()


def optimizationRFPerMonth(valid):
    valid = valid.sort_values("Month")
    reg = joblib.load("../model/RandomForestRegressor.pkl")
    y_valid = valid["Sales"]
    x_valid = valid.drop("Sales", axis=1)
    x_valid, y_valid = preprocessMM(x_valid, y_valid)
    print(reg.score(x_valid, y_valid))
    y_hat = reg.predict(x_valid)
    y_hat = np.array(y_hat)
    overall_correction_factor = 0.934600
    w_month = {}
    prev = 0
    for month in range(1, 13):
        curr = x_valid[x_valid["Month"] == month].shape[0] + prev
        w = 0.90
        ws = []
        errors = []
        while w < 1.1:
            y_hat_temp = y_hat[prev:curr]
            # print(x_valid.head())
            y_hat_temp = y_hat_temp * overall_correction_factor * w
            ws.append(w)
            error = LossFuction.basicRmspe(y_valid.values[prev:curr], y_hat_temp)
            errors.append(error)
            w += 0.0001
            # print("w = %f, error = %f" % (w, error))
        min_error = min(errors)
        w = ws[np.argmin(errors)]
        print("*" * 30)
        print("min:w = %f, error = %f" % (w, min_error))
        w_month[month] = w
        prev = curr
    y_hat_overall = y_hat* overall_correction_factor
    error = LossFuction.basicRmspe(y_valid.values, y_hat_overall)
    print("apply overall correction factor, error: ", error)
    prev = 0

    for month in range(1, 13):
        curr = x_valid[x_valid["Month"] == month].shape[0] + prev
        y_hat[prev:curr] = y_hat[prev:curr] * w_month[month] * overall_correction_factor
        prev = curr
    error = LossFuction.basicRmspe(y_valid.values, y_hat)
    print("after applying factors for every months, error: ", error)
    plt.plot(y_hat_overall[::200])
    plt.plot(y_hat[::200])
    plt.plot(y_valid.iloc[::200].values)
    plt.legend(["apply overall correction factor", "after applying factors for every months", "real"])
    plt.show()


def optimizationRFParam(x_train, y_train, x_valid, y_valid):
    # reg = RandomForestRegressor()
    # reg.fit(x_train, y_train)
    # print(reg.score(x_valid,y_valid))
    # x_train = x_train[x_train["Year"] == 2015].iloc[:10000]
    # x_valid = x_valid[x_valid["Year"] == 2015].iloc[:10000]
    param = {"n_estimators": range(150, 200, 10),
             "min_samples_leaf": range(1, 10, 2)
             }
    t1 = time.time()
    reg = RandomForestRegressor(n_jobs=-1)
    gs = GridSearchCV(estimator=reg, param_grid=param)
    gs.fit(x_train, y_train)
    t2 = time.time()
    print("best score: %f, best param: " % gs.best_score_, gs.best_params_)
    print("time is ", t2 - t1)
