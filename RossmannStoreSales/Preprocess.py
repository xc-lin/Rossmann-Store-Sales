import pandas
from sklearn.preprocessing import MinMaxScaler


def preprocess(x_train, y_train):
    one_hot_code_features = ["DayOfWeek", "StateHoliday",  "StoreType", "Assortment",
                             "Year", "Month", "Day","Promo2SinceWeek",
                             "Promo2SinceYear"]
    x_train = pandas.get_dummies(x_train, columns=one_hot_code_features)
    x_train["CompetitionDistance"] = (x_train["CompetitionDistance"] - x_train["CompetitionDistance"].min()) / (
                x_train["CompetitionDistance"].max() - x_train["CompetitionDistance"].min())
    # print(x_train.head())
    # mm = MinMaxScaler()
    # # data_CompetitionDistance = train_data[["CompetitionDistance"]]
    # scalered_dis = mm.fit(x_train)
    # scalered_dis = pandas.DataFrame(scalered_dis, columns=["CompetitionDistance"])

    # XgboostModel.xgboostModel(x_train, y_train, x_valid, y_valid)
    # x_train = pandas.concat([one_hot_part, scalered_dis], axis=1)
    return x_train, y_train
def preprocessMM(x_train, y_train):
    x_train["CompetitionDistance"] = (x_train["CompetitionDistance"] - x_train["CompetitionDistance"].min()) / (
            x_train["CompetitionDistance"].max() - x_train["CompetitionDistance"].min())

    return x_train, y_train