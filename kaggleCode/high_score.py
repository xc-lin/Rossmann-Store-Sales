import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

import datetime
import calendar


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


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
store = pd.read_csv('../input/store.csv')

train.Store.nunique() == store.Store.nunique()

df = train.merge(store, how='left', left_on=train.Store, right_on=store.Store)
df.drop(['key_0', 'Store_y'], axis=1, inplace=True)
df = df.rename(columns={'Store_x': 'Store'})
df.shape

df.Date = pd.to_datetime(df.Date)
df['Day'] = df.Date.dt.day
df['Month'] = df.Date.dt.month
df['Year'] = df.Date.dt.year

features_x = ['Store', 'Date', 'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday', 'StateHoliday']
features_y = ['SalesLog']

train['is_train'] = 1
test['is_train'] = 0
df = pd.concat([train, test])

df.Date = pd.to_datetime(df.Date)  # Converting date to required format

df = df.loc[~((df['Open'] == 1) & (df['Sales'] == 0))]  # Removing rows with Sales 0

df.StateHoliday = df.StateHoliday.map({0: '0', 'a': 'a', 'b': 'b', 'c': 'c', '0': '0'})  # mixed data types
# aaa
df.StateHoliday = LabelEncoder().fit_transform(df.StateHoliday)  # Encoding for XG Boost

var_name = 'Date'
# aaa
df[var_name + 'Day'] = df[var_name].dt.day  # addding day
df[var_name + 'Week'] = df[var_name].dt.week  # adding week
df[var_name + 'Month'] = df[var_name].dt.month  # adding month
df[var_name + 'Year'] = df[var_name].dt.year  # adding year
df[var_name + 'DayOfYear'] = df[var_name].dt.dayofyear  # adding dayofyear

features_x.remove(var_name)  # removing Date
features_x.append(var_name + 'Day')
features_x.append(var_name + 'Week')
features_x.append(var_name + 'Month')
features_x.append(var_name + 'Year')
features_x.append(var_name + 'DayOfYear')

store.StoreType = LabelEncoder().fit_transform(store.StoreType)  # encoding StoreType
store.Assortment = LabelEncoder().fit_transform(store.Assortment)  # encoding Assortment



# Most Promos are done on DayofWeek 4
df['DaysTillMaxPromo'] = df.DayOfWeek.apply(lambda x: 4 - x)

df['PromoTomorrow'] = df.Promo.shift(-1)
df['PromoYesterday'] = df.Promo.shift(1)

promo_features = ['DaysTillMaxPromo', 'PromoTomorrow', 'PromoYesterday']

features_x = list(set(features_x + promo_features))

df.Sales = df.Sales.apply(lambda x: np.nan if x == 0 else x)  # Convert 0 to NaNs

df.loc[df['is_train'] == 1, 'SalesLog'] = np.log(
    1 + df.loc[df['is_train'] == 1]['Sales'])  # Transforming Sales to 1+log


import xgboost as xgb

data = df.loc[(df['is_train'] == 1) & (df['Open'] == 1) ]
x_train, x_test, y_train, y_test = train_test_split(data[features_x],
                                                    data[features_y],
                                                    test_size=0.1,
                                                    random_state=42)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(x_train.info())
print(x_test.info())
dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test, y_test)

num_round = 7000
evallist = [(dtrain, 'train'), (dtest, 'test')]

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
from xgboost import plot_importance

plot_importance(model)
plt.show()

submit = df.loc[df['is_train'] == 0]
dsubmit = xgb.DMatrix(submit[features_x])
predictions = model.predict(dsubmit)

df_predictions = submit['Id'].reset_index()
df_predictions['Id'] = df_predictions['Id'].astype('int')
df_predictions['Sales'] = (np.exp(predictions) - 1) * 0.985  # Scale Back

df_predictions.sort_values('Id', inplace=True)
df_predictions[['Id', 'Sales']].to_csv('solution.csv', index=False)
