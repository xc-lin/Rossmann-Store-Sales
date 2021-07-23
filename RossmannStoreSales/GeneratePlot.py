from matplotlib import pyplot as plt
import seaborn as sns


def generatePlot(train_data):
    plt.subplots(figsize=(30, 25))
    sns.heatmap(train_data.corr(), cmap="YlGnBu", annot=True, vmin=-0.1, vmax=0.1, center=0)
    sns.pairplot(train_data[0:100])

    store_sales = train_data.groupby("Store", as_index=False)["Sales"].mean()
    sns.boxplot(store_sales["Sales"])
    sns.displot(store_sales, x="Sales")
    plt.show()

    sales_per_year = train_data.groupby("Year", as_index=False)[["Sales"]].mean()

    plt.subplot(3, 1, 1)
    plt.title("Average sale of every year")
    sns.barplot(data=sales_per_year, x="Year", y="Sales")
    plt.xlabel("Year")
    plt.ylabel("Sales")

    sales_per_month = train_data.groupby("Month", as_index=False)[["Sales"]].mean()
    plt.subplot(3, 1, 2)
    plt.title("Average sale of every month")
    plt.plot(sales_per_month["Month"], sales_per_month["Sales"], "o-")
    plt.xlabel("Month")
    plt.ylabel("Sales")

    sales_per_day = train_data.groupby("Day", as_index=False)[["Sales"]].mean()
    plt.subplot(3, 1, 3)
    plt.title("Average sale of every day")
    plt.plot(sales_per_day["Day"], sales_per_day["Sales"], "o-")
    plt.xlabel("Day")
    plt.ylabel("Sales")
    plt.show()

    sales_per_month_year = train_data.groupby(["Month", "Year"], as_index=False)[["Sales"]].mean()
    plt.title("Average sale of every month in different year")
    sns.pointplot(data=sales_per_month_year, x="Month", y="Sales", hue="Year")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.show()

    a, (sub1, sub2) = plt.subplots(1, 2, figsize=(16, 8))
    sales_store_type = train_data.groupby("StoreType", as_index=False)[["Sales"]].mean()
    plt.title("Average sale of every StoreType")
    sns.barplot(data=sales_store_type, x="StoreType", y="Sales", ax=sub1)

    sales_assortment = train_data.groupby("Assortment", as_index=False)[["Sales"]].mean()
    plt.title("Average sale of every Assortment")
    sns.barplot(data=sales_assortment, x="Assortment", y="Sales", ax=sub2)
    plt.show()

    competition_distance_sales = train_data.groupby("CompetitionDistance", as_index=False)[["Sales"]].mean()
    plt.plot(competition_distance_sales["CompetitionDistance"], competition_distance_sales["Sales"], "-")
    plt.xlabel("CompetitionDistance")
    plt.ylabel("Sales")
    plt.show()

    competition_open_since_month_sales = \
        train_data[train_data["CompetitionOpenSinceMonth"] != 0].groupby("CompetitionOpenSinceMonth", as_index=False)[
            ["Sales"]].mean()
    plt.plot(competition_open_since_month_sales["CompetitionOpenSinceMonth"],
             competition_open_since_month_sales["Sales"],
             "-")
    plt.xlabel("CompetitionOpenSinceMonth")
    plt.ylabel("Sales")
    plt.show()

    open_sales = train_data.groupby("Open", as_index=False)[["Sales"]].mean()
    sns.barplot(data=open_sales, x="Open", y="Sales")
    plt.xlabel("Open")
    plt.ylabel("Sales")
    plt.show()

    sns.boxplot(data=train_data, x="Promo", y="Sales")
    plt.show()

    a, (sub1, sub2) = plt.subplots(1, 2, figsize=(20, 8))
    sns.scatterplot(data=train_data, x="Customers", y="Sales", hue="Promo", ax=sub1)
    sns.scatterplot(data=train_data, x="Customers", y="Sales", hue="Promo2", ax=sub2)
    plt.show()

    a, (sub1, sub2) = plt.subplots(1, 2, figsize=(20, 8))

    promo_train = train_data.groupby("Promo", as_index=False)["Sales"].mean()
    sns.barplot(data=promo_train, x="Promo", y="Sales", ax=sub1)

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

    school_holiday_sale_cus = train_data.groupby("SchoolHoliday", as_index=False)[["Sales", "Customers"]].mean()
    sns.barplot(data=school_holiday_sale_cus, x="SchoolHoliday", y="Sales", ax=sub1[0])
    sns.barplot(data=school_holiday_sale_cus, x="SchoolHoliday", y="Customers", ax=sub1[1])

    state_holiday_sale_cus = train_data.groupby("StateHoliday", as_index=False)[["Sales", "Customers"]].mean()
    sns.barplot(data=state_holiday_sale_cus, x="StateHoliday", y="Sales", ax=sub2[0])
    sns.barplot(data=state_holiday_sale_cus, x="StateHoliday", y="Customers", ax=sub2[1])
    plt.show()

    customers_sales = train_data.groupby("Customers", as_index=False)["Sales"].mean()
    sns.scatterplot(data=customers_sales, x="Customers", y="Sales")
    plt.show()

    promo_sales = train_data[train_data["Store"] == 30].groupby("IsInPromo", as_index=False)["Sales"].mean()

    sns.barplot(data=promo_sales, x="IsInPromo", y="Sales")
    plt.show()
    sns.scatterplot(data=train_data, x="Customers", y="Sales", hue="IsInPromo")
    plt.show()
