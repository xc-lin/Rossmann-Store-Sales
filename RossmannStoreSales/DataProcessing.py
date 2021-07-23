import joblib
import numpy as np
import pandas
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import plot_importance

from RossmannStoreSales import Tree


def handleZero(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1. / (y[ind])
    return w


def basicRmspe(y, y_hat):
    one_over_y = handleZero(y)
    s1 = one_over_y * (y - y_hat) ** 2
    s2 = np.mean(s1)
    result = np.sqrt(s2)
    return result


def rmspe(y_hat, y):
    y = y.get_label()
    y = np.exp(y) - 1
    y_hat = np.exp(y_hat) - 1
    one_over_y = handleZero(y)
    s1 = one_over_y * (y - y_hat)
    s2 = np.mean(s1 ** 2)
    result = np.sqrt(s2)
    return "rmspe", result



pandas.set_option("display.max_columns", 1000)
pandas.set_option("display.max_rows", 1000)
store_data = pandas.read_csv("../input/store.csv")
train_data = pandas.read_csv("../input/train.csv")
test_data = pandas.read_csv("../input/test.csv")
print("-" * 20, "train_data", "-" * 20)
print(train_data.info())
print()

print("-" * 20, "store_data", "-" * 20)
print(store_data.info())
print(store_data.isnull().sum())
print()

print("-" * 20, "test_data", "-" * 20)
print(test_data.info())
print(test_data[test_data.isnull().T.any()])
print("-" * 100)

store_data.fillna(0, inplace=True)
store_data["PromoInterval"] = store_data["PromoInterval"].apply(lambda x: "" if x == 0 else x)

# the data in train data of store 622 is open except 7
# print(train_data.loc[train_data["Store"] == 622][["DayOfWeek", "Open"]])
null_data = test_data.isnull().T.any()

# set
test_data["Open"][null_data] = test_data["DayOfWeek"][null_data].apply(lambda x: 1 if x != 7 else 0)

train_data = pandas.merge(train_data, store_data, on="Store")
test_data = pandas.merge(test_data, store_data, on="Store")
# train_data=train_data[train_data["Sales"]>0]
train_data = train_data.loc[~((train_data['Open'] == 1) & (train_data['Sales'] == 0))]


def dataProcess(data):
    #  Convert date to year, month, and day
    data["Year"] = data["Date"].apply(lambda x: int(x.split("-")[0]))
    monthNumToWord = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", \
                      7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec"}
    data["Month"] = data["Date"].apply(lambda x: int(x.split("-")[1]))

    data["Month"] = data["Month"].map(monthNumToWord)
    data["Day"] = data["Date"].apply(lambda x: int(x.split("-")[2]))
    #  Convert PromoInterval to whether this month is in the promotion date
    data["IsInPromo"] = data.apply(lambda x: 1 if x["Month"] in x["PromoInterval"] else 0, axis=1)
    #  Convert the Month to int
    data["Month"] = data["Date"].apply(lambda x: int(x.split("-")[1]))
    # add the feature of Week of year and day of year
    data["Date"] = pandas.to_datetime(data["Date"])
    data["WeekOfYear"] = data["Date"].dt.week
    data["DayOfYear"] = data["Date"].dt.dayofyear
    #
    # print(train_data.head())
    letterToNum = {"0": 0, "a": 1, "b": 2, "c": 3, "d": 4}
    # train_data = train_data[train_data["Sales"] > 0]
    data["StoreType"] = data["StoreType"].map(letterToNum)
    data["Assortment"] = data["Assortment"].map(letterToNum)
    data["StateHoliday"] = data["StateHoliday"].apply(lambda x: "0" if x == 0 else x)
    data["StateHoliday"] = data["StateHoliday"].map(letterToNum).astype(int)
    data["Store"] = data["Store"].astype(int)
    data["Promo2"] = data["Promo2"].astype(int)
    data["Promo2SinceWeek"] = data["Promo2SinceWeek"].astype(int)
    data["Promo2SinceYear"] = data["Promo2SinceYear"].astype(int)
    data["CompetitionOpenSinceMonth"] = data["CompetitionOpenSinceMonth"].astype(int)
    data["CompetitionOpenSinceYear"] = data["CompetitionOpenSinceYear"].astype(int)
    data["Open"] = data["Open"].astype(int)


dataProcess(train_data)
dataProcess(test_data)

'''
plt.subplots(figsize=(30, 25))
sns.heatmap(train_data.corr(), cmap="YlGnBu", annot=True, vmin=-0.1, vmax=0.1, center=0)
sns.pairplot(train_data[0:100])


store_sales = train_data.groupby("Store", as_index=False)["Sales"].mean()
sns.boxplot(store_sales["Sales"])
sns.displot(store_sales, x="Sales")
plt.show()


salesPerYear = train_data.groupby("Year", as_index=False)[["Sales"]].mean()

plt.subplot(3, 1, 1)
plt.title("Average sale of every year")
sns.barplot(data=salesPerYear,x="Year", y="Sales")
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

salesPerMonthYear = train_data.groupby(["Month", "Year"], as_index=False)[["Sales"]].mean()
plt.title("Average sale of every month in different year")
sns.pointplot(data=salesPerMonthYear, x="Month", y="Sales", hue="Year")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()


a, (sub1, sub2) = plt.subplots(1, 2, figsize=(16, 8))
Sales_StoreType = train_data.groupby("StoreType", as_index=False)[["Sales"]].mean()
plt.title("Average sale of every StoreType")
sns.barplot(data=Sales_StoreType,x="StoreType", y="Sales", ax=sub1)

Sales_Assortment = train_data.groupby("Assortment", as_index=False)[["Sales"]].mean()
plt.title("Average sale of every Assortment")
sns.barplot(data=Sales_Assortment,x="Assortment", y="Sales", ax=sub2)
plt.show()

competitionDistance_Sales = train_data.groupby("CompetitionDistance", as_index=False)[["Sales"]].mean()
plt.plot(competitionDistance_Sales["CompetitionDistance"], competitionDistance_Sales["Sales"], "-")
plt.xlabel("CompetitionDistance")
plt.ylabel("Sales")
plt.show()

competitionOpenSinceMonth_Sales = \
    train_data[train_data["CompetitionOpenSinceMonth"] != 0].groupby("CompetitionOpenSinceMonth", as_index=False)[
        ["Sales"]].mean()
plt.plot(competitionOpenSinceMonth_Sales["CompetitionOpenSinceMonth"], competitionOpenSinceMonth_Sales["Sales"], "-")
plt.xlabel("CompetitionOpenSinceMonth")
plt.ylabel("Sales")
plt.show()




open_Sales = train_data.groupby("Open", as_index=False)[["Sales"]].mean()
sns.barplot(data=open_Sales, x="Open", y="Sales")
plt.xlabel("Open")
plt.ylabel("Sales")
plt.show()



sns.boxplot(data=train_data, x="Promo", y="Sales")
plt.show()

a, (sub1, sub2) = plt.subplots(1, 2, figsize=(20, 8))
sns.scatterplot(data=train_data,x="Customers", y="Sales", hue="Promo", ax=sub1)
sns.scatterplot(data=train_data,x="Customers", y="Sales", hue="Promo2", ax=sub2)
plt.show()

a, (sub1, sub2) = plt.subplots(1, 2, figsize=(20, 8))

promo_train = train_data.groupby("Promo", as_index=False)["Sales"].mean()
sns.barplot(data=promo_train,x="Promo", y="Sales", ax=sub1)


promo2_train = train_data.groupby("Promo2", as_index=False)["Sales"].mean()
sns.barplot(data=promo2_train, x="Promo2", y="Sales", ax=sub2)
plt.show()

sales_of_weekday = train_data.groupby("DayOfWeek", as_index=False)["Sales"].mean()

sns.pointplot(data=sales_of_weekday, x="DayOfWeek", y="Sales", markers="o")
plt.show()

a, (sub1, sub2) = plt.subplots(1, 2, figsize=(20, 8))
sns.scatterplot(data=train_data, x="Customers", y="Sales", hue="SchoolHoliday", ax=sub1)
sns.scatterplot(data=train_data, x="Customers", y="Sales", hue="StateHoliday", ax=sub2)
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

extractedFeatures = ["Store", "WeekOfYear", "DayOfYear", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday",
                     "StoreType",
                     "Assortment", "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                     "Promo2",
                     "IsInPromo", "Year", "Month", "Day", "Open", "Promo2SinceWeek", "Promo2SinceYear"]

# train_data=train_data[train_data["Sales"]>0]
x_train = train_data[extractedFeatures]
y_train = train_data["Sales"]

features = extractedFeatures.copy()
features.append("Sales")
train = train_data[features]
train, valid = train_test_split(train, test_size=0.1, random_state=42)
# valid, test = train_test_split(valid, test_size=0.3, random_state=15)

y_train_v = train[["Sales"]]
x_train_v = train.drop("Sales", axis=1)

y_valid = valid[["Sales"]]
x_valid = valid.drop("Sales", axis=1)

# y_test = test[["Sales"]]
# x_test = test.drop("Sales", axis=1)

# LinearRegressionModel.linearRegression(x_train, y_train)


#
# LinearRegressionModel.sgdRegression(x_train, y_train)

#
# alphas = []
# alpha = 0
# mean_scores = []
# while alpha < 100:
#     alphas.append(alpha)
#     mean_score = LinearRegressionModel.ridgeRegression(x_train, y_train, alpha=alpha)
#     print("alpha: ", alpha)
#     alpha += 5
#     mean_scores.append(mean_score)
# maxV = max(mean_scores)
# index = mean_scores.index(maxV)
# print("Ridge alpha: %.1f mean score: %f" % (alphas[index], maxV))
# plt.plot(alphas, mean_scores)
# plt.xlabel("alpha")
# plt.ylabel("mean score")
# plt.show()
#
#
# alphas = []
# alpha = 0
# mean_scores = []
# while alpha < 20:
#     alphas.append(alpha)
#     mean_score = LinearRegressionModel.lassoRegression(x_train, y_train, alpha=alpha)
#     print("alpha: ", alpha)
#     alpha += 0.2
#
#     mean_scores.append(mean_score)
#
# maxV = max(mean_scores)
# index = mean_scores.index(maxV)
# print("Lasso alpha: %.1f mean score: %f" % (alphas[index], maxV))
# plt.plot(alphas, mean_scores)
# plt.xlabel("alpha")
# plt.ylabel("mean score")
# plt.show()


Tree.decisionTree(x_train_v, y_train_v)
# Ensemble.randomForest(x_train_v, y_train_v, 10)
# Ensemble.extraTrees(x_train, y_train)
# #
# Ensemble.gradientBoosting(x_train, y_train)

#

#
# LinearRegressionModel.generateLinearRegression(x_train_v, y_train_v, x_valid, y_valid)
# LinearRegressionModel.generateSGDRegression(x_train_v, y_train_v, x_valid, y_valid)
# LinearRegressionModel.generateRidgeRegression(x_train_v, y_train_v, 37, x_valid, y_valid)
# LinearRegressionModel.generateLassoRegression(x_train_v, y_train_v, 0.2, x_valid, y_valid)
# Tree.generateDecisionTree(x_train_v, y_train_v, x_valid, y_valid)
# Ensemble.generateRandomForest(x_train_v, y_train_v, x_valid, y_valid)
# Ensemble.generateExtraTrees(x_train_v, y_train_v, x_valid, y_valid)
# Ensemble.generateGradientBoosting(x_train_v, y_train_v, x_valid, y_valid)
# print(x_train.columns)
# models = ["RandomForestRegressor","ExtraTreesRegressor", "DecisionTreeRegressor"]
# for model in models:
#     optimization(x_train_v, y_train_v, model)
# optimizationRFTotal(valid)
# optimizationRFPerMonth(valid)
#
# train_opt = train[train["Year"] == 2015].iloc[:10000]
# valid_opt = valid[valid["Year"] == 2015].iloc[:1000]
# y_train_opt = train["Sales"]
# x_train_opt = train.drop("Sales", axis=1)
# # valid = valid[valid["Sales"] > 0]
# y_valid_opt = valid["Sales"]
# x_valid_opt = valid.drop("Sales", axis=1)
# optimizationRFParam(x_train_opt, y_train_opt, x_valid_opt, y_valid_opt)
# print(test_data.info())
#


#
# extractedFeatures.remove("Sales")
# print(extractedFeatures)
# test_data = test_data.sort_values("Id")
# store_ids = test_data["Id"]
# test_data = test_data[extractedFeatures]
# reg = joblib.load("../model/RandomForestRegressor.pkl")
# factor = 0.943
# for i in range(10):
#     print(factor)
#     test_data, a = preprocessMM(test_data, a)
#     Y_pred = reg.predict(test_data) * factor
#
#     submission = {
#         "Id": store_ids,
#         "Sales": Y_pred
#     }
#
#     # submission = Series(Y_pred, index=store_ids)
#     submission = pandas.DataFrame(submission)
#     submission.to_csv('submission%f.csv' % factor, index=False)
#     factor += 0.0001


#
# extractedFeatures.remove("Sales")
# print(extractedFeatures)
# test_data = test_data.sort_values("Month")
# store_ids = test_data["Id"]
# test_data = test_data[extractedFeatures]
# test_data, a = preprocessMM(test_data, a)
# # reg = joblib.load("../model/RandomForestRegressor.pkl")
#
# a = 0
# Y_pred = optimizationRFPerMonthSubmission(valid, test_data)
#
# submission = {
#     "Id": store_ids,
#     "Sales": Y_pred
# }
#
# # submission = Series(Y_pred, index=store_ids)
# submission = pandas.DataFrame(submission)
# submission.to_csv('submission.csv', index=False)

# Xgboost
# xgboost(x_train_v, y_train_v, x_valid, y_valid, test_data,extractedFeatures)
#
#
# print(extractedFeatures)
#
# y_train_v = np.log(1 + y_train_v)
# y_valid = np.log(1 + y_valid)
# # y_test = np.log(1 + y_test)
#
#
# train_matrix = xgb.DMatrix(x_train_v, y_train_v)
# valid_matrix = xgb.DMatrix(x_valid, y_valid)
# # dtest = xgb.DMatrix(x_test, y_test)
# print()
# num_round = 10000
# evallist = [(train_matrix, 'train'), (valid_matrix, 'valid')]
#
# param = {'max_depth': 9,
#          'eta': 0.06,
#          'subsample': 0.75,
#          'colsample_bytree': 0.6,
#          'objective': 'reg:squarederror', }
#
# plst = list(param.items())
# print(123)
# # y_train = np.log(1 + y_train)
# # dtrain2 = xgb.DMatrix(x_train, y_train)
# # res = xgb.cv(plst, dtrain2, num_round, feval=rmspe, verbose_eval=1, early_stopping_rounds=2, nfold=2)
# # print(res)
# # reg = xgb.train(plst, train_matrix, num_round, evallist,
# #                 feval=rmspe, verbose_eval=1, early_stopping_rounds=50)
# # joblib.dump(reg, '../model/Xgboost.pkl')
# factor = 0.975
# reg = joblib.load("../model/Xgboost.pkl")
# print(123)
#
# y_valid_hat = reg.predict(valid_matrix)
#
# plt.title("Xgboost predict data after correction")
# print(y_valid.values[0:10])
# plt.plot((np.exp(y_valid_hat[::900]) - 1)*factor)
# plt.plot(np.exp(y_valid.values[::900]) - 1)
# plt.legend(["y_hat", "real"])
# plt.show()
#
#
# # Print Feature Importance
#
# plt.subplots(figsize=(30, 10))
# plot_importance(reg)
# plt.show()
#
#
# submit = test_data
# dsubmit = xgb.DMatrix(submit[extractedFeatures])
# predictions = reg.predict(dsubmit)
#
# df_predictions = submit['Id'].reset_index()
# df_predictions['Id'] = df_predictions['Id'].astype('int')
#
#
# df_predictions['Sales'] = (np.exp(predictions) - 1) * factor  # Scale Back
# df_predictions.sort_values('Id', inplace=True)
# df_predictions[['Id', 'Sales']].to_csv('solution.csv', index=False)
