# 导入需要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from time import time
import pickle

# 导入数据集
store = pd.read_csv('../input/store.csv')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# 1.检查数据# 可以看前几行观察下数据的基本情况
print(store.head())
print(train.head())
print(test.head())
# 对缺失数据检查 检查训练集数据有无缺失print(train.isnull().sum())
# 检查测试集数据有无缺失
print(test.isnull().sum())
# 发现open有11行缺失值# 看看缺哪些，通过查看train里622号店的营业情况发现，622号店周一到周六都是营业的
print(test[test['Open'].isnull()])  # 所以我们认为缺失的部分是应该正常营业的，用1填充
test.fillna(1, inplace=True)
# 接下来看看store的缺失值,store的缺失值比较多
print(store.isnull().sum())
# 下面对缺失数据进行填充# 店铺竞争数据缺失，而且缺失的都是对应的。原因不明，而且数量也比较多，如果用中值或均值来填充，有失偏颇。暂且填0，解释意义就是刚开业
# 店铺促销信息的缺失是因为没有参加促销活动，所以我们以0填充
store.fillna(0, inplace=True)
# 分析店铺销量随时间的变化
strain = train[train['Sales'] > 0]
strain.loc[strain['Store'] == 1, ['Date', 'Sales']].plot(x='Date', y='Sales', figsize=(16, 4))
plt.show()

# 合并数据
# 我们只需要销售额大于0的数据
train = train[train['Sales'] > 0]
# 把store基本信息合并到训练和测试数据集上
train = pd.merge(train, store, on='Store', how='left')
test = pd.merge(test, store, on='Store', how='left')
print(train.info())

for data in [train, test]:
    # 将时间特征进行拆分和转化
    data['year'] = data['Date'].apply(lambda x: x.split('-')[0])
    data['year'] = data['year'].astype(int)
    data['month'] = data['Date'].apply(lambda x: x.split('-')[1])
    data['month'] = data['month'].astype(int)
    data['day'] = data['Date'].apply(lambda x: x.split('-')[2])
    data['day'] = data['day'].astype(int)
    # 将'PromoInterval'特征转化为'IsPromoMonth'特征，表示某天某店铺是否处于促销月，1表示是，0表示否
    # 提示下：这里尽量不要用循环，用这种广播的形式，会快很多。循环可能会让你等的想哭
    month2str = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct',
                 11: 'Nov', 12: 'Dec'}
    data['monthstr'] = data['month'].map(month2str)
    data['IsPromoMonth'] = data.apply(
        lambda x: 0 if x['PromoInterval'] == 0 else 1 if x['monthstr'] in x['PromoInterval'] else 0, axis=1)
    # 将存在其它字符表示分类的特征转化为数字
    mappings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
    data['StoreType'].replace(mappings, inplace=True)
    data['Assortment'].replace(mappings, inplace=True)
    data['StateHoliday'].replace(mappings, inplace=True)
# 删掉训练和测试数据集中不需要的特征
df_train = train.drop(['Date', 'Customers', 'Open', 'PromoInterval', 'monthstr'], axis=1)
df_test = test.drop(['Id', 'Date', 'Open', 'PromoInterval', 'monthstr'], axis=1)
# 如上所述，保留训练集中最近六周的数据用于后续模型的测试
Xtrain = df_train[6 * 7 * 1115:]
Xtest = df_train[:6 * 7 * 1115]
# 大家从表上可以看下相关性
plt.subplots(figsize=(24, 20))
sns.heatmap(df_train.corr(), cmap='RdYlGn', annot=True, vmin=-0.1, vmax=0.1, center=0)
sns.pairplot(df_train[0:1000])
plt.show()

# 提取后续模型训练的数据集# 拆分特征与标签，并将标签取对数处理
ytrain = np.log1p(Xtrain['Sales'])
ytest = np.log1p(Xtest['Sales'])
Xtrain = Xtrain.drop(['Sales'], axis=1)
Xtest = Xtest.drop(['Sales'], axis=1)


# 定义评价函数，可以传入后面模型中替代模型本身的损失函数
def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat / y - 1) ** 2))


def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return 'rmspe', rmspe(y, yhat)


# 初始模型构建# 参数设定
params = {'objective': 'reg:linear', 'booster': 'gbtree', 'eta': 0.03, 'max_depth': 10, 'subsample': 0.9,
          'colsample_bytree': 0.7, 'silent': 1, 'seed': 10}
num_boost_round = 6000
dtrain = xgb.DMatrix(Xtrain, ytrain)
dvalid = xgb.DMatrix(Xtest, ytest)
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

# 模型训练
print('Train a XGBoost model')
start = time()
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=100, feval=rmspe_xg,
                verbose_eval=True)
pickle.dump(gbm, open("pima.pickle.dat", "wb"))
end = time()
print('Train time is {:.2f} s.'.format(end - start))
'''Train time is 923.86 s.训练花费15分钟。。'''

gbm = pickle.load(open("pima.pickle.dat", "rb"))
# 采用保留数据集进行检测
print('validating')
Xtest.sort_index(inplace=True)
ytest.sort_index(inplace=True)
yhat = gbm.predict(xgb.DMatrix(Xtest))
error = rmspe(np.expm1(ytest), np.expm1(yhat))
print('RMSPE: {:.6f}'.format(error))
'''validatingRMSPE: 0.128683'''
# 构建保留数据集预测结果
res = pd.DataFrame(data=ytest)
res['Predicition'] = yhat
res = pd.merge(Xtest, res, left_index=True, right_index=True)
res['Ratio'] = res['Predicition'] / res['Sales']
res['Error'] = abs(res['Ratio'] - 1)
res['Weight'] = res['Sales'] / res['Predicition']
res.head()
# 分析保留数据集中任意三个店铺的预测结果
col_1 = ['Sales', 'Prediction']
col_2 = ['Ratio']
L = np.random.randint(low=1, high=1115, size=3)
print('Mean Ratio of prediction and real sales data is {}:store all'.format(res['Ratio'].mean()))
for i in L:
    s1 = pd.DataFrame(res[res['Store'] == i], columns=col_1)
    s2 = pd.DataFrame(res[res['Store'] == i], columns=col_2)
    s1.plot(s1.format(i), figsize = (12, 4))
    s2.plot(s2.format(i), figsize = (12, 4))
    print('Mean Ratio of prediction and real sales data is {}:store {}'.format(s2['Ratio'].mean(), i))
# 分析偏差最大的10个预测结果
res.sort_values(['Error'], ascending=False, inplace=True)
print(res[:10])
# 从分析结果来看，初始模型已经可以比较好的预测保留数据集的销售趋势，但相对真实值，模型的预测值整体要偏高一些。从对偏差数据分析来看，偏差最大的3个数据也是明显偏高。因此，我们可以以保留数据集为标准对模型进行偏差校正。
