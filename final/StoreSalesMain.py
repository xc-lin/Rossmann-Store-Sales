import argparse

import pandas
from sklearn.model_selection import train_test_split

import Model
from GeneratePlot import generatePlot
import warnings


# warnings.filterwarnings("ignore")


def handleMissingValue(train_data, store_data, test_data):
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
    return train_data, test_data


def dataProcess(data):
    #  Convert date to year, month, and day
    data["Year"] = data["Date"].apply(lambda x: int(x.split("-")[0]))
    monthNumToWord = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
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
    data["WeekOfYear"] = data["Date"].dt.isocalendar().week.astype(int)
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


def extractFeatures(train_data, test_data):
    extractedFeatures = ["Store", "WeekOfYear", "DayOfYear", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday",
                         "StoreType", "Assortment", "CompetitionDistance", "CompetitionOpenSinceMonth",
                         "CompetitionOpenSinceYear", "Promo2", "IsInPromo", "Year", "Month", "Day", "Open",
                         "Promo2SinceWeek", "Promo2SinceYear"]

    # train_data=train_data[train_data["Sales"]>0]
    # x_train = train_data[extractedFeatures]
    # y_train = train_data["Sales"]

    features = extractedFeatures.copy()
    features.append("Sales")
    train = train_data[features]
    train, valid = train_test_split(train, test_size=0.1, random_state=42)
    # valid, test = train_test_split(valid, test_size=0.3, random_state=15)

    y_train_v = train[["Sales"]]
    x_train_v = train.drop("Sales", axis=1)

    y_valid = valid[["Sales"]]
    x_valid = valid.drop("Sales", axis=1)
    test_features = extractedFeatures.copy()
    test_features.append("Id")
    test_data = test_data[test_features]
    return x_train_v, y_train_v, x_valid, y_valid, test_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true', default=False, help="to generate some plots to analyse data")
    parser.add_argument('--model', type=str, default="xgboost", help="linear or decisionTree or extraTrees, "
                                                                     "gradientBoosting or randomForest or xgboost")
    parser.add_argument('--predict', action='store_true', default=False,
                        help='predict the test data and generate submission.csv by generated xgboost model directly')
    parser.add_argument('--nfolds', type=int, default="10", help="Number of folds. Must be at least 2 default:10")

    args = parser.parse_args()
    nfolds = args.nfolds
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
    print()
    print("-" * 20, "test_data", "-" * 20)
    print(test_data.info())
    print("-" * 20, "nan data of test_data", "-" * 20)
    print(test_data[test_data.isnull().T.any()])
    print("-" * 100)

    print("-" * 20, "Start processing data...", "-" * 20)
    train_data, test_data = handleMissingValue(train_data, store_data, test_data)
    print("processing train data...")
    dataProcess(train_data)
    print("processing test data...")
    dataProcess(test_data)
    print("-" * 20, "data process finished", "-" * 20)

    if args.plot:
        print("-" * 20, "Start generating plots...", "-" * 20)
        generatePlot(train_data)
        print("-" * 20, "plotsGenerating finished", "-" * 20)

    x_train_v, y_train_v, x_valid, y_valid, test_data = extractFeatures(train_data, test_data)
    if args.predict:
        print("-" * 20, "Starting generating submission.csv and plots of predict data of xgboost...", "-" * 20)
        Model.xgboostPredict(x_valid, y_valid, test_data)
        print("-" * 20, "generation is finished", "-" * 20)
    elif args.model == "linear":
        print("-" * 20, "Starting testing LinearRegression...", "-" * 20)
        Model.linearRegression(x_train_v, y_train_v, x_valid, y_valid, nfolds)
        print("-" * 20, "LinearRegression test is finished", "-" * 20)
    # elif args.model == "ridge":
    #     print("-" * 20, "Starting testing ExtraTrees...", "-" * 20)
    #     Model.ridge(x_train_v, y_train_v, x_valid, y_valid, nfolds)
    #     print("-" * 20, "ExtraTrees test is finished", "-" * 20)

    elif args.model == "decisionTree":
        print("-" * 20, "Starting testing DecisionTree...", "-" * 20)
        Model.decisionTree(x_train_v, y_train_v, x_valid, y_valid, nfolds)
        print("-" * 20, "DecisionTree test is finished", "-" * 20)

    elif args.model == "extraTrees":
        print("-" * 20, "Starting testing ExtraTrees...", "-" * 20)
        Model.extraTrees(x_train_v, y_train_v, x_valid, y_valid, nfolds)
        print("-" * 20, "ExtraTrees test is finished", "-" * 20)

    elif args.model == "gradientBoosting":
        print("-" * 20, "Starting testing GradientBoosting...", "-" * 20)
        Model.gradientBoosting(x_train_v, y_train_v, x_valid, y_valid, nfolds)
        print("-" * 20, "GradientBoosting test is finished", "-" * 20)

    elif args.model == "randomForest":
        print("-" * 20, "Starting testing GradientBoosting...", "-" * 20)
        Model.randomForest(x_train_v, y_train_v, x_valid, y_valid, nfolds)
        print("-" * 20, "GradientBoosting test is finished", "-" * 20)

    elif args.model == "xgboost":
        print("-" * 20, "Starting testing xgboost...", "-" * 20)
        Model.xgboost(x_train_v, y_train_v, x_valid, y_valid, test_data)
        print("-" * 20, "xgboost test is finished", "-" * 20)


if __name__ == '__main__':
    main()
