import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
from xgboost import plot_importance


# Thanks for providing this in the forum
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1. / (y[ind] ** 2)
    return w


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return rmspe


def rmspe_xg(yhat, y):
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return "rmspe", rmspe


def xgboost(x_train_v, y_train_v, x_valid, y_valid, test_data, extractedFeatures):
    # x_train, y_train = preprocessMM(x_train, y_train)
    # x_valid, y_valid = preprocessMM(x_valid, y_valid)

    y_train_v = np.log(1 + y_train_v)
    y_valid = np.log(1 + y_valid)

    print(x_train_v.info())
    print(y_train_v.info())
    print(x_valid.info())
    print(y_valid.info())
    dtrain = xgb.DMatrix(x_train_v, y_train_v)
    dvalid = xgb.DMatrix(x_valid, y_valid)
    print()
    num_round = 7000
    evallist = [(dtrain, 'train'), (dvalid, 'valid')]

    param = {'max_depth': 9,
             'eta': 0.06,
             'subsample': 0.75,
             'colsample_bytree': 0.6,
             'objective': 'reg:squarederror', }

    plst = list(param.items())
    print(123)
    model = xgb.train(plst, dtrain, num_round, evallist,
                      feval=rmspe_xg, verbose_eval=1, early_stopping_rounds=100)
    print(123)
    # Print Feature Importance
    plt.figure(figsize=(18, 8))

    plot_importance(model)
    plt.show()
    a = 0
    # test_data, a = preprocessMM(test_data, a)
    submit = test_data
    dsubmit = xgb.DMatrix(submit[extractedFeatures])
    predictions = model.predict(dsubmit)

    df_predictions = submit['Id'].reset_index()
    df_predictions['Id'] = df_predictions['Id'].astype('int')
    df_predictions['Sales'] = (np.exp(predictions) - 1) * 0.985  # Scale Back

    df_predictions.sort_values('Id', inplace=True)
    df_predictions[['Id', 'Sales']].to_csv('solution3.csv', index=False)
