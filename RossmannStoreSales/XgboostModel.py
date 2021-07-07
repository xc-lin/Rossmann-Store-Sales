import numpy as np
import xgboost as xgb

from RossmannStoreSales import LossFuction


def xgboostModel(x_train, y_train, x_valid, y_valid):


    params = {"objective": "reg:linear",
              "booster": "gbtree",
              "eta": 0.3,
              "max_depth": 10,
              "subsample": 0.9,
              "colsample_bytree": 0.7,
              "silent": 1,
              "seed": 1301
              }
    num_boost_round = 300

    xgb_train = xgb.DMatrix(x_train, np.log1p(y_train))
    xgb_valid = xgb.DMatrix(x_valid, np.log1p(y_valid))
    watchlist = [(xgb_train, 'train'), (xgb_valid, 'eval')]
    gbm = xgb.train(params, xgb_train, num_boost_round, evals=watchlist, \
                    early_stopping_rounds=100, feval=LossFuction.rmspe, verbose_eval=True)

    y_hat = gbm.predict(xgb_valid)
    error = LossFuction.basicRmspe(y_valid, np.expm1(y_hat))
    print('RMSPE: {:.6f}'.format(error))

    print(np.expm1(y_hat))
    print(y_valid.values)
