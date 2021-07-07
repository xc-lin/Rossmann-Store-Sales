import math

import numpy as np
import pandas
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import XgboostModel as xgb
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

from RossmannStoreSales import XgboostModel
from RossmannStoreSales import LinearRegressionModel

pandas.set_option("display.max_columns", 1000)
pandas.set_option("display.max_rows", 1000)
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
store_data[store_data["PromoInterval"].isnull()] = ""
store_data.fillna(0, inplace=True)
print("------test_data-------")
print(test_data.info())
print(test_data.head())
print(test_data[test_data.isnull().T.any()])
# the data in train data of store 622 is open except 7
# print(train_data.loc[train_data['Store'] == 622][["DayOfWeek", "Open"]])
null_data = test_data.isnull().T.any()
# set
test_data["Open"][null_data] = (test_data["DayOfWeek"][null_data] != 7).astype(int)

train_data = train_data[train_data["Sales"] > 0]

train_data = pandas.merge(train_data, store_data, on="Store")

# so that we can compute the corr easily
train_data["Year"] = train_data["Date"].apply(lambda x: int(x.split('-')[0]))
monthNumToWord = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', \
                  7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
train_data["Month"] = train_data["Date"].apply(lambda x: int(x.split('-')[1]))
train_data["Month"] = train_data["Month"].map(monthNumToWord)
train_data["Day"] = train_data["Date"].apply(lambda x: int(x.split('-')[2]))
train_data["IsInPromo"] = train_data.apply(lambda x: 1 if x["Month"] in x["PromoInterval"] else 0, axis=1)
train_data["Month"] = train_data["Date"].apply(lambda x: int(x.split('-')[1]))
# print(train_data.head())
letterToNum = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
train_data["StoreType"] = train_data["StoreType"].map(letterToNum)
train_data["Assortment"] = train_data["Assortment"].map(letterToNum)
train_data["StateHoliday"] = train_data["StateHoliday"].apply(lambda x: '0' if x == 0 else x)
train_data["StateHoliday"] = train_data["StateHoliday"].map(letterToNum).astype(int)

print(train_data.info())
# train_data = train_data.drop("Promo2SinceWeek", axis=1)
# train_data = train_data.drop("Promo2", axis=1)
# train_data = train_data.drop("Promo2SinceYear", axis=1)
'''
plt.subplots(figsize=(30, 25))
sns.heatmap(train_data.corr(), cmap='YlGnBu', annot=True, vmin=-0.1, vmax=0.1, center=0)
sns.pairplot(train_data[0:100])

salesPerYear = train_data.groupby("Year", as_index=False)[["Sales"]].mean()

plt.subplot(3, 1, 1)
plt.title("Average sale of every year")
sns.barplot(salesPerYear["Year"], salesPerYear["Sales"])
plt.xlabel("Year")
plt.ylabel("Sales")

salesPerMonth = train_data.groupby("Month", as_index=False)[["Sales"]].mean()
plt.subplot(3, 1, 2)
plt.title("Average sale of every month")
plt.plot(salesPerMonth["Month"], salesPerMonth["Sales"], "o-")
plt.xlabel("Month")
plt.ylabel("Sales")

salesPerDay = train_data.groupby("Day", as_index=False)[["Sales"]].mean()
plt.subplot(3, 1, 3)
plt.title("Average sale of every day")
plt.plot(salesPerDay["Day"], salesPerDay["Sales"], "o-")
plt.xlabel("Day")
plt.ylabel("Sales")
plt.show()

a, (sub1, sub2) = plt.subplots(1, 2, figsize=(16, 8))
Sales_StoreType = train_data.groupby("StoreType", as_index=False)[["Sales"]].mean()
plt.title("Average sale of every StoreType")
sns.barplot(Sales_StoreType["StoreType"], Sales_StoreType["Sales"], ax=sub1)

Sales_Assortment = train_data.groupby("Assortment", as_index=False)[["Sales"]].mean()
plt.title("Average sale of every Assortment")
sns.barplot(Sales_Assortment["Assortment"], Sales_Assortment["Sales"], ax=sub2)
plt.show()

competitionDistance_Sales = train_data.groupby("CompetitionDistance", as_index=False)[["Sales"]].mean()
plt.plot(competitionDistance_Sales["CompetitionDistance"], competitionDistance_Sales["Sales"], "-")
plt.xlabel("CompetitionDistance")
plt.ylabel("Sales")
plt.show()


a, (sub1, sub2) = plt.subplots(1, 2, figsize=(20, 8))
sns.scatterplot(train_data["Customers"], train_data["Sales"], hue=train_data["Promo"], ax=sub1)
sns.scatterplot(train_data["Customers"], train_data["Sales"], hue=train_data["Promo2"], ax=sub2)
plt.show()

a, (sub1, sub2) = plt.subplots(1, 2, figsize=(20, 8))

promo_train = train_data.groupby("Promo", as_index=False)["Sales"].mean()
sns.barplot(data=promo_train,x="Promo", y="Sales", ax=sub1)


promo2_train = train_data.groupby("Promo2", as_index=False)["Sales"].mean()
sns.barplot(data=promo2_train, x="Promo2", y="Sales", ax=sub2)
plt.show()


sales_of_weekday = train_data.groupby("DayOfWeek", as_index=False)["Sales"].mean()

sns.lineplot(data=sales_of_weekday, x="DayOfWeek", y="Sales", markers="o")
plt.show()

a, (sub1, sub2) = plt.subplots(1, 2, figsize=(20, 8))
sns.scatterplot(train_data, x="Customers", y="Sales", hue=train_data["SchoolHoliday"], ax=sub1)
sns.scatterplot(train_data, x="Customers", y="Sales", hue=train_data["StateHoliday"], ax=sub2)
plt.show()


a, (sub1, sub2) = plt.subplots(2, 2, figsize=(20, 20))

SchoolHoliday_sale_cus = train_data.groupby("SchoolHoliday", as_index=False)[["Sales", "Customers"]].mean()
sns.barplot(data=SchoolHoliday_sale_cus, x="SchoolHoliday", y="Sales", ax=sub1[0])
sns.barplot(data=SchoolHoliday_sale_cus, x="SchoolHoliday", y="Customers", ax=sub1[1])

StateHoliday_sale_cus = train_data.groupby("StateHoliday", as_index=False)[["Sales", "Customers"]].mean()
sns.barplot(data=StateHoliday_sale_cus, x="StateHoliday", y="Sales", ax=sub2[0])
sns.barplot(data=StateHoliday_sale_cus, x="StateHoliday", y="Customers", ax=sub2[1])
plt.show()

customers_sales = train_data.groupby("Customers", as_index=False)["Sales"].mean()
sns.scatterplot(data=customers_sales, x="Customers", y="Sales")
plt.show()

promo_sales = train_data[train_data["Store"] == 30].groupby("IsInPromo", as_index=False)["Sales"].mean()

sns.barplot(data=promo_sales, x="IsInPromo", y="Sales")
plt.show()
sns.scatterplot(data=train_data, x="Customers", y="Sales", hue="IsInPromo")
plt.show()
'''

extractedFeatures = ["Store", "Sales", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday", "StoreType", "Assortment",
                     "CompetitionDistance", "Promo2", "IsInPromo", "Year", "Month", "Day"]

train_data["Store"] = train_data["Store"].astype(int)
train_data["CompetitionDistance"] = train_data["CompetitionDistance"].astype(int)
train_data["Promo2"] = train_data["Promo2"].astype(int)

# train = train_data[extractedFeatures]

ss = StandardScaler()

train, valid = train_test_split(train_data[extractedFeatures], test_size=0.012, random_state=10)

x_train = train.drop(["Sales"], axis=1)
# x_train = ss.fit_transform(x_train)
y_train = train["Sales"]
x_valid = valid.drop("Sales", axis=1)
# x_valid = ss.fit_transform(x_valid)
y_valid = valid["Sales"]

# XgboostModel.xgboostModel(x_train, y_train, x_valid, y_valid)
# LinearRegressionModel.linearRegression(x_train, y_train, x_valid, y_valid)
# alpha=0
# while alpha<50:
#     LinearRegressionModel.ridgeRegression(x_train, y_train, x_valid, y_valid,alpha=alpha)
#     alpha+=0.1
train_data=train_data[extractedFeatures]
rossmann_dic = dict(list(train_data.groupby('Store')))
test_dic = dict(list(test_data.groupby('Store')))

for i in rossmann_dic:
    store = rossmann_dic[i]

    # define training and testing sets
    X_train = store.drop(["Sales", "Store"], axis=1)
    Y_train = store["Sales"]


    # Linear Regression
    lreg = LinearRegression()
    lreg.fit(X_train, Y_train)

    print(lreg.score(X_train, Y_train))

