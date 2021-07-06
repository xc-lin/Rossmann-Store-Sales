import pandas
import seaborn as sns
from matplotlib import pyplot as plt

pandas.set_option("display.max_columns", 1000)
store_data = pandas.read_csv("../input/store.csv")
train_data = pandas.read_csv("../input/train.csv")
test_data = pandas.read_csv("../input/test.csv")
print("------train_data------")
print(train_data.info())
print(train_data.head())
print()

print("------store_data------")
print(store_data.info())
print(store_data.head())
print(store_data.isnull().sum())
print()
store_data.fillna(0, inplace=True)
print("------test_data-------")
print(test_data.info())
print(test_data.head())
print(test_data[test_data.isnull().T.any()])
# the data in train data of store 622 is open except 7
print(train_data.loc[train_data['Store'] == 622][["DayOfWeek", "Open"]])
null_data = test_data.isnull().T.any()
# set
test_data["Open"][null_data] = (test_data["DayOfWeek"][null_data] != 7).astype(int)

train_data = pandas.merge(train_data, store_data, on="Store")

# so that we can compute the corr easily
train_data["Year"] = train_data["Date"].apply(lambda x: x.split('-')[0])
train_data["Month"] = train_data["Date"].apply(lambda x: x.split('-')[1])
train_data["Day"] = train_data["Date"].apply(lambda x: x.split('-')[2])
train_data = train_data.drop("Date", axis=1)

letterToNum = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
train_data["StoreType"] = train_data["StoreType"].map(letterToNum)
train_data["Assortment"] = train_data["Assortment"].map(letterToNum)
print(train_data.info())
train_data = train_data.drop("Promo2SinceWeek", axis=1)
train_data = train_data.drop("Promo2", axis=1)
train_data = train_data.drop("Promo2SinceYear", axis=1)

plt.subplots(figsize=(24, 20))
sns.heatmap(train_data.corr(), cmap='YlGnBu', annot=True, vmin=-0.1, vmax=0.1, center=0)
sns.pairplot(train_data[0:100])
plt.show()
