import numpy as np
import Xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold

from RossmannStoreSales import LossFuction


def xgboostModel(x_train, y_train):
    params = {"objective": "reg:linear",
              "booster": "gbtree",
              "eta": 0.1,
              "max_depth": 10,
              "subsample": 0.9,
              "colsample_bytree": 0.7,
              "silent": 1,
              "seed": 1301
              }
    num_boost_round = 300

    xgb_train = xgb.DMatrix(x_train, np.log1p(y_train))
    watchlist = [(xgb_train, 'train')]
    # gbm = xgb.train(params, xgb_train, num_boost_round, evals=watchlist, \
    #                 early_stopping_rounds=100, feval=LossFuction.rmspe, verbose_eval=True)
    # xgb.plot_importance(gbm)
    # plt.show()
    # y_hat = gbm.predict(xgb_valid)
    # error = LossFuction.basicRmspe(y_valid, np.expm1(y_hat))
    # print('RMSPE: {:.6f}'.format(error))
    #
    # print(np.expm1(y_hat))
    # print(y_valid.values)
    res = xgb.cv(params, xgb_train, num_boost_round, \
                 early_stopping_rounds=100, feval=LossFuction.rmspe, verbose_eval=5, nfold=10,metrics='auc')
    print(res)