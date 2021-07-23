import joblib
import numpy as np
import pandas
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import plot_importance









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


# Tree.decisionTree(x_train_v, y_train_v)
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


print(extractedFeatures)

y_train_v = np.log(1 + y_train_v)
y_valid = np.log(1 + y_valid)
# y_test = np.log(1 + y_test)


train_matrix = xgb.DMatrix(x_train_v, y_train_v)
valid_matrix = xgb.DMatrix(x_valid, y_valid)
# dtest = xgb.DMatrix(x_test, y_test)
print()
num_round = 10000
evallist = [(train_matrix, 'train'), (valid_matrix, 'valid')]

param = {'max_depth': 9,
         'eta': 0.06,
         'subsample': 0.75,
         'colsample_bytree': 0.6,
         'objective': 'reg:squarederror', }

plst = list(param.items())
print(123)
# y_train = np.log(1 + y_train)
# dtrain2 = xgb.DMatrix(x_train, y_train)
# res = xgb.cv(plst, dtrain2, num_round, feval=rmspe, verbose_eval=1, early_stopping_rounds=2, nfold=2)
# print(res)
reg = xgb.train(plst, train_matrix, num_round, evallist,
                feval=rmspe, verbose_eval=1, early_stopping_rounds=50)
joblib.dump(reg, '../model/Xgboost.pkl')
factor = 0.975
reg = joblib.load("../model/Xgboost.pkl")
print(123)
y_valid_hat = reg.predict(valid_matrix)

plt.title("Xgboost predict data after correction")
plt.plot((np.exp(y_valid_hat[::900]) - 1)*factor)
plt.plot(np.exp(y_valid.values[::900]) - 1)
plt.legend(["y_hat", "real"])
plt.show()


# Print Feature Importance

plt.subplots(figsize=(30, 10))
plot_importance(reg)
plt.show()


submit = test_data
dsubmit = xgb.DMatrix(submit[extractedFeatures])
predictions = reg.predict(dsubmit)

df_predictions = submit['Id'].reset_index()
df_predictions['Id'] = df_predictions['Id'].astype('int')


df_predictions['Sales'] = (np.exp(predictions) - 1) * factor  # Scale Back
df_predictions.sort_values('Id', inplace=True)
df_predictions[['Id', 'Sales']].to_csv('solution.csv', index=False)
