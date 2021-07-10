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
    if model == "DecisionTreeRegressor":
        x_train, y_train = preprocess(x_train, y_train)
        # dot_data = tree.export_graphviz(decision_tree=reg, max_depth=5, feature_names=x_train.columns,
        #                                 filled=True, rounded=True, out_file=None)
        # graph = pydotplus.graph_from_dot_data(dot_data)
        # graph.write_pdf("tree.pdf")
    importance_metrics = pandas.Series(reg.feature_importances_, index=x_train.columns).sort_values()
    plt.figure(figsize=(14, 7))
    # plt.style.use('ggplot')
    importance_metrics[importance_metrics > 0.01].plot.barh()  # 帅选重要程度超过0.005的
    plt.xlabel('importance')
    plt.ylabel('feature')
    plt.title('%s feature importance' % model)
    plt.show()


def optimizationRF(x_valid, y_valid):
    reg = joblib.load("../model/RandomForestRegressor.pkl")
    x_valid, y_valid = preprocessMM(x_valid, y_valid)
    w = 0.95
    ws = []
    errors = []
    while w < 1:
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


def optimizationRFParam(x_train, y_train,x_valid,y_valid):
    reg = RandomForestRegressor()
    reg.fit(x_train, y_train)
    print(reg.score(x_valid,y_valid))
    # param = {"n_estimators": range(100, 101)}
    # t1 = time.time()
    # reg = RandomForestRegressor()
    # gs = GridSearchCV(estimator=reg, param_grid=param)
    # gs.fit(x_train, y_train)
    # t2 = time.time()
    # print("best score: %f, best param: " % gs.best_score_, gs.best_params_)
    # print("time is ", t2 - t1)
