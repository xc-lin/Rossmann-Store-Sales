import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 1000)

def open_csv(csv_name):
    with open("../input/{}".format(csv_name)) as f:
        data = pd.read_csv(f)
    return data


data_store = open_csv('store.csv')
data_train = open_csv('train.csv')
data_test = open_csv('test.csv')


def check_file(data, data_name):
    print('{:*^30}'.format('{}前3行').format(data_name))
    print(data.head(3))
    print('{:*^30}'.format('{}各行信息').format(data_name))
    print(data.info(), '\n', '\n')


datas = [data_store, data_train, data_test]
data_names = ['data_store', 'data_train', 'data_test']
for data, data_name in zip(datas, data_names):
    check_file(data, data_name)


def check_unique(data):
    for column in data.columns:
        print('{}为一值数量:'.format(column), len(data[column].unique()))


# 检查store表
check_unique(data_store)
np.sort(data_store['CompetitionOpenSinceMonth'].unique())
np.sort(data_store['CompetitionOpenSinceYear'].unique())
check_unique(data_train)
check_unique(data_test)
data_train.StateHoliday.unique()
data_train.StateHoliday.unique()
data_test.StateHoliday.unique()
data_test.Open.unique()


def check_none(data):
    print('{:*^30}'.format('有空值的列'))
    print(data.isnull().any(axis=0))
    print('{:*^30}'.format('有空值的行数量'))
    print(data.isnull().any(axis=1).sum())


check_none(data_store)


def check_none_together(data, columns):
    indexes = []
    indexes_len = []
    for column in columns:
        index = data[data[column].isnull()].index  # 找出每列有空值的行索引
        indexes_len.append(len(index))  # 算出每列缺失行数量
        indexes.append(index)  # 将每列缺失行索引保留
    if set(indexes[0]) == set(indexes[1]) and set(indexes[1]) == set(indexes[2]):  # 检查每列缺失的行索引是否都一样
        print('相同行都为空值')
    else:
        print(indexes_len[0], indexes_len[1], indexes_len[2])  # 不一样，则把每个列缺失行数打印出来
    return indexes


# 对竞争信息为空的验证
index_1 = check_none_together(data_store,
                              ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'])

index_2 = check_none_together(data_store,
                              ['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'])

data_store.iloc[index_1[0]]

data_store.iloc[index_1[1]].sample(5)

indexes_2 = check_none_together(data_store, ['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval'])
sum(data_store.iloc[indexes_2[0]]['Promo2'] != 0)  # 检验空值行的Promo2是否都为0

data_train_new = pd.merge(data_train, data_store, on='Store', how='left')
data_train_new.head()

data_train_new.Date = pd.to_datetime(data_train_new.Date)
data_train_new.info()
'''
df1=data_train_new.groupby('Store')['Sales'].sum()/10000
sns.distplot(df1)
plt.title('每家店sales总额的分布')
plt.xlabel('销售总额/万元')

df2=data_train_new.groupby('DayOfWeek')['Sales'].mean().reset_index()
sns.pointplot(data=df2,x='DayOfWeek',y='Sales')
plt.title('一周不同的天平均每天每个店的销售情况')
Text(0.5,1,'一周不同的天平均每天每个店的销售情况')

#一周的平均营业率
df3=data_train_new.groupby('DayOfWeek')['Open'].mean().reset_index().rename(columns={'Open':'open_rate'})
sns.pointplot(data=df3,x='DayOfWeek',y='open_rate')
plt.title('一周的平均营业率')
Text(0.5,1,'一周的平均营业率')

#探索每个月平均销售情况
df4=data_train_new.groupby('Date')['Sales'].mean().resample('M',kind='period').mean().reset_index()#resample能够把聚合时间变成月份，去除日
fig=plt.figure(figsize=(20,10))
ax=fig.add_subplot(1,1,1)
sns.pointplot(data=df4,x='Date',y='Sales',ax=ax)
plt.xticks(rotation=45)
ax.set_title('每月平均销售额情况')
Text(0.5,1,'每月平均销售额情况')

data_train_new['year']=data_train_new.Date.map(lambda x:x.year)
data_train_new['month']=data_train_new.Date.map(lambda x:x.month)
data_train_new['day']=data_train_new.Date.map(lambda x:x.day)
data_train_new.head()

#不同年每个月份平均销售情况
df5=data_train_new.Sales.groupby([data_train_new.year,data_train_new.month]).mean().reset_index()
fig=plt.figure(figsize=(10,5))
ax=fig.add_subplot(1,1,1)
sns.pointplot(x='month',y='Sales',hue='year',data=df5,ax=ax)
ax.set_title('不同年每个月份平均销售情况')
plt.legend(loc='best')


#不同月份每天平均销售情况
df6=data_train_new.Sales.groupby([data_train_new.month,data_train_new.day]).mean().reset_index()
fig=plt.figure(figsize=(20,10))
ax=fig.add_subplot(1,1,1)
sns.pointplot(x='day',y='Sales',hue='month',data=df6,ax=ax)
ax.set_title('不同月份每天平均销售情况')

#标记12月份最高值与最低值
dec_max=df6[df6.month==12].Sales.max()
dec_min=df6[df6.month==12].Sales.min()
plt.annotate('12月份最高值',xy=(15,dec_max),xytext=(17,dec_max),arrowprops={'facecolor':'b'})#xy中横坐标第一个数为1，所以填写的横坐标和图上的差一
plt.annotate('12月份最低值',xy=(24,dec_min),xytext=(24,dec_min+1000),arrowprops={'facecolor':'b'})
plt.legend(loc='best')

#探索竞争者距离和销售额关系
df7=data_train_new.groupby('Store')['CompetitionDistance','Sales'].agg({'CompetitionDistance':'mean','Sales':'sum'})#每个店销售总额
df7['CompetitionDistance']=df7['CompetitionDistance'].map(lambda x:np.log(x))#对距离取log，收缩分布
df7['Sales']=df7['Sales'].map(lambda x:x/10000)#销售额以万元为单位
fig=plt.figure(figsize=(10,5))
ax=fig.add_subplot(1,1,1)
sns.regplot(x='CompetitionDistance',y='Sales',data=df7,ax=ax)
ax.set_title('竞争者距离和销售额关系')
plt.legend(loc='best')

#店铺促销次数比例分布
df7=data_train_new.groupby('Store')['Promo'].sum()
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
sns.distplot(df7,ax=ax)

df8=df7.value_counts()
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
sns.barplot(df8.index,df8.values,ax=ax)
ax.set(**{'title':'店铺促销次数分布'})
[Text(0.5,1,'店铺促销次数分布')]

#对比促销天和非促销天销售分布情况
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
sns.boxplot(y='Sales',x='Promo',data=data_train_new,ax=ax)
ax.set(**{'title':'促销天和非促销天销售分布情况'})
[Text(0.5,1,'促销天和非促销天销售分布情况')]

#对比促销天和非促销天销售分布情况(一天销售额超过20000的)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
sns.boxplot(y='Sales',x='Promo',data=data_train_new[data_train_new.Sales>=20000],ax=ax)
ax.set(**{'title':'促销天和非促销天销售分布情况'})
[Text(0.5,1,'促销天和非促销天销售分布情况')]

#有无promo2的店数量对比
df9=data_store.Promo2.value_counts()
df9.plot(kind='bar')

#有无promo2的每天每个店销售情况对比
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
sns.boxplot(y='Sales',x='Promo2',data=data_train_new,ax=ax)
ax.set(**{'title':'有无promo2的每天每个店销售情况对比'})


#有无promo2的店铺的平均每天销售情况对比
df10=data_train_new.groupby('Store')['Promo2','Sales'].mean()
fig=plt.figure(figsize=(12,5))
ax=fig.subplots(1,2)
sns.boxplot(y='Sales',x='Promo2',data=df10,ax=ax[0])
ax[0].set(**{'title':'有无promo2的店铺的平均每天销售情况对比'})
sns.boxplot(y='Sales',x='Promo2',data=df10[df10.Sales>=10000],ax=ax[1])
ax[1].set(**{'title':'有无promo2的店铺的平均每天销售情况对比(平均销售额大于10000的店)'})

#对比stateholiday销售分布情况
fig,ax=plt.subplots(1,3,sharey=True,figsize=(10,5))

#不是国家节假日的销售分布
sns.violinplot(y='Sales',x='StateHoliday',data=data_train_new[data_train_new.StateHoliday=='无'],ax=ax[0])
ax[0].set(**{'title':'不是国家节假日的销售分布'})
#国家节假日的销售分布
sns.violinplot(y='Sales',x='StateHoliday',data=data_train_new[data_train_new.StateHoliday!='无'],ax=ax[2])
ax[2].set(**{'title':'国家节假日的销售分布'})
#国家节假日的销售分布(营业的店)
sns.violinplot(y='Sales',x='StateHoliday',data=data_train_new[(data_train_new.StateHoliday!='无')&(data_train_new.Sales>0)],ax=ax[1])
ax[1].set(**{'title':'国家节假日的销售分布(营业的店)'})

#对比SchoolHoliday销售分布情况

fig,ax=plt.subplots(1,2,figsize=(10,5))
#SchoolHoliday的销售分布
sns.boxplot(y='Sales',x='SchoolHoliday',data=data_train_new,ax=ax[0])
ax[0].set(**{'title':'SchoolHoliday的销售分布'})
#SchoolHoliday的销售分布（Sales>=15000）
sns.boxplot(y='Sales',x='SchoolHoliday',data=data_train_new[data_train_new.Sales>=15000],ax=ax[1])
ax[1].set(**{'title':'SchoolHoliday的销售分布（Sales>=15000）'})

#StoreType
fig,ax=plt.subplots(1,2,figsize=(10,5))
#StoreType的销售分布
sns.boxplot(y='Sales',x='StoreType',data=data_train_new,ax=ax[0])
ax[0].set(**{'title':'StoreType的销售分布'})
sns.barplot(y='Sales',x='StoreType',data=data_train_new,ax=ax[1])
ax[1].set(**{'title':'不同StoreType的平均每天每个商店销售分布'})

fig,ax=plt.subplots(1,2,figsize=(10,5))
#Assortment的销售分布
sns.boxplot(y='Sales',x='Assortment',data=data_train_new,ax=ax[0])
ax[0].set(**{'title':'Assortment的销售分布'})
sns.barplot(y='Sales',x='Assortment',data=data_train_new,ax=ax[1])
ax[1].set(**{'title':'不同Assortment的平均每天每个商店销售分布'})

#顾客数与销售额的关系
fig=plt.figure(figsize=(10,5))
ax=fig.add_subplot(1,1,1)
data_train_new.plot.scatter(y='Sales',x='Customers',ax=ax)
ax.set(**{'title':'顾客数与销售额的关系'})

#探索是否营业与销售额关系
fig=plt.figure(figsize=(10,5))
ax=fig.add_subplot(1,1,1)
sns.boxplot(y='Sales',x='Open',data=data_train_new,ax=ax)
ax.set(**{'title':'是否营业与销售额关系'})
'''

data_train.head(2)
data_test.head(2)
# 删除顾客数一列
data_train = data_train.drop('Customers', axis=1)
# 对data_test先做和data_train之前相同的变化
data_test.StateHoliday = data_test.StateHoliday.map({'0': '无', 0: '无', 'a': '公共假日', 'b': '复活节假期', 'c': '圣诞节'})
# 连接test表和train表
data_train_test = pd.concat([data_train, data_test.drop('Id', axis=1)], axis=0, ignore_index=True)

data_train_test.sample(5)

# 连接store表和train_test表
data_train_test = pd.merge(data_train_test, data_store, on='Store', how='left')
data_train_test.head(3)

data_train_test.Date = pd.to_datetime(data_train_test.Date)
data_train_test.info()

# 把日期时间分解
data_train_test['year'] = data_train_test.Date.map(lambda x: x.year)
data_train_test['month'] = data_train_test.Date.map(lambda x: x.month)
data_train_test['day'] = data_train_test.Date.map(lambda x: x.day)
data_train_test['Date'] = data_train_test.Date.map(lambda x: x.date())
# 检查空值
check_none(data_train_test)

# 检查open列空值情况
sum(data_train_test.Open.isnull())

data_train_test[data_train_test.Open.isnull()]

data_train_test.loc[[1017688, 1018544, 1019400, 1020256], 'Open'] = 1
data_train_test.loc[[1017688, 1018544, 1019400, 1020256]]  # 检查是否改正成功

date_null = data_train_test[data_train_test.Open.isnull()].Date
# 查看这个日期下其他店铺是否营业情况
data_date_null = data_train_test[data_train_test.Date.isin(date_null)]
df12 = data_date_null.groupby(['Date', 'Open'])['Store'].count().rename('count').reset_index()

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1)
sns.barplot('Date', 'count', hue='Open', data=df12, ax=ax)
# ax.set_xticklabels(df12.Date, rotation=45)

index_null = data_train_test[data_train_test.Open.isnull()].index
data_train_test.loc[index_null, 'Open'] = 1
sum(data_train_test.Open.isnull())  # 查看是否补全空值


# 检查每列的空值行数
def check_none_col(data, column):
    print('{}的缺失行数：'.format(column), sum(data[column].isnull()))
    print('{}的缺失率：'.format(column), round(sum(data[column].isnull()) / data.shape[0], 3))


columns = ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
           'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
for i in columns:
    check_none_col(data_store, i)
data_tem = pd.concat([data_train_test[columns], data_train_test.Sales], axis=1)
data_tem.corr()

# 备份数据，以防改错数据
data_safe = data_train_test
# 平均值填补距离
data_train_test.CompetitionDistance = data_train_test.CompetitionDistance.fillna(
    data_train_test.CompetitionDistance.mean())
# 众数填补其他
for i in columns:
    data_train_test[i] = data_train_test[i].fillna(data_train_test[i].mode()[0])  # mode返回series，不是一个数
check_none(data_train_test)


# 浮点数转为整数
def convert_to_int(data, columns):
    for i in columns:
        data[i] = data[i].astype(np.int64)
    return data


columns_float = ['Open', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']
data_train_test = convert_to_int(data_train_test, columns_float)


# 将数值数据转化为分类数据
def convert_to_object(data, columns):
    for i in columns:
        data[i] = data[i].astype(np.str)
    return data


columns_classify = ['DayOfWeek', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek',
                    'Promo2SinceYear', 'year', 'month', 'day']
data_train_test = convert_to_object(data_train_test, columns_classify)
check_unique(data_train_test)  # 该函数见上篇数据处理部分
data_store['PromoInterval'].value_counts()


# 分类数据二值化处理
def convert_to_twovalues(data, columns):
    connect_column = []
    for i in columns:
        df_tem = pd.get_dummies(data[i], prefix=i)
        connect_column.append(df_tem)
    data_new = pd.concat(connect_column, axis=1)
    return data_new


columns_value_processing = ['DayOfWeek', 'StateHoliday', 'StoreType', 'Assortment', 'CompetitionOpenSinceMonth',
                            'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'year',
                            'month', 'day']
data_value_processing = convert_to_twovalues(data_train_test, columns_value_processing)
data_value_processing.head()


# 对一行多个值进行二值化处理,使用每行值是多个值的
# 之前以为‘PromoInterval’需要这样处理所以做了这个函数，本次项目不需要，可保留作为以后项目备用
def convert_to_twovalues_more(data, column):
    month = []
    for i in data[column]:
        month.extend(i.split(','))  # 注意要把字符串变成列表
    index_list = pd.Index(list(set(month)))
    print(index_list)
    data_new = pd.DataFrame(np.zeros([data.shape[0], len(index_list)]), columns=index_list)
    for i, data in enumerate(data[column]):
        column_position = index_list.get_indexer(data.split(','))
        data_new.iloc[i, column_position] = 1
    return data_new


# 直接运行速度很慢，所以现在data_store修改再连接,此代码留到以后使用

# 数值类型进行归一化(0-1范围)
def feature_standarize(data, columns):
    combine_col = []
    for j in columns:
        min_ = data[j].min()
        max_ = data[j].max()
        standard_col = data[j].apply(lambda x: (x - min_) / (max_ - min_))  # 标准化为0-1范围
        # standard_col=data[j].apply(lambda x:(x-data[j].min())/(data[j].max()-data[j].min()))这样运行很慢，因为每次都要找min，max
        combine_col.append(standard_col)
    data_new2 = pd.concat(combine_col, axis=1)
    return data_new2


data_CompetitionDistance = feature_standarize(data_train_test, ['CompetitionDistance'])
data_CompetitionDistance.head()

# 连接数据data_value_processing，data_CompetitionDistance，data_train_test未作修改的列

data_train_test_new = pd.concat(
    [data_value_processing, data_CompetitionDistance, data_train_test[['Open', 'Promo', 'SchoolHoliday', 'Promo2']]],
    axis=1)
data_train_test_new.info()

# 将数据拆分为训练_测试数据和预测数据
index_split = data_train.shape[0] - 1
data_train_test_final = pd.concat([data_train_test_new.loc[:index_split], data_train['Sales']], axis=1)
data_for_predict = data_train_test_new.loc[data_train.shape[0]:]
print(data_train_test_final.info())  # 打印训练测试数据信息
print(data_for_predict.info())  # 打印预测数据信息

# 导入模型
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

# 拆分数据为训练数据和测试数据
data_x = data_train_test_final.iloc[:, :-1]
data_y = data_train_test_final.iloc[:, -1:]
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

# 用cross_val_score交叉检验各模型的评分
lr_model = LinearRegression()
tree_model = DecisionTreeRegressor()
gbdt_model = GradientBoostingRegressor()
rfr_model = RandomForestRegressor()
# models = [lr_model, tree_model, rfr_model, gbdt_model]
# model_names = ['lr_model', 'tree_model', 'rfr_model', 'gbdt_model']
models = [tree_model]
model_names = ['rfr_model']
scores = []
for model, model_name in zip(models, model_names):
    t5 = time.time()
    score = cross_val_score(model, train_x, train_y, cv=StratifiedKFold(5))
    t6 = time.time()
    print('{}运行时间：'.format(model_name), (t6 - t5))
    scores.append(score)
score_matrix = pd.DataFrame(scores, index=model_names)
score_matrix['mean'] = score_matrix.mean(axis=1)
score_matrix['std'] = score_matrix.std(axis=1)
print('{:*^30}'.format('各模型分数矩阵'))
print(score_matrix)
