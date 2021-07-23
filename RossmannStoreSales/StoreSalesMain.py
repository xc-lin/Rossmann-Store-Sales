import argparse

import pandas

from RossmannStoreSales.GeneratePlot import generatePlot


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
    data["WeekOfYear"] = data["Date"].dt.isocalendar().week
    data["DayOfYear"] = data["Date"].dt.isocalendar().year
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', type=bool, default=False, help="True or False, Whether to generate plots")
    parser.add_argument('--model', type=str, default="xgboost", help="linear, decisionTree, extraTrees, "
                                                                     "gradientBoosting, xgboost")
    parser.add_argument('--predict', type=bool, default=False,
                        help='True or False, Whether to predict the test data and generate submission.csv')
    args = parser.parse_args()

    pandas.set_option("display.max_columns", 1000)
    pandas.set_option("display.max_rows", 1000)
    store_data = pandas.read_csv("input/store.csv")
    train_data = pandas.read_csv("input/train.csv")
    test_data = pandas.read_csv("input/test.csv")
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

    print("-" * 20, "Start process data", "-" * 20)
    train_data, test_data = handleMissingValue(train_data, store_data, test_data)

    dataProcess(train_data)
    dataProcess(test_data)
    print("-" * 20, "data process finished", "-" * 20)
    if args.plot:
        (train_data)


if __name__ == '__main__':
    main()
