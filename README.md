# **Rossmann-Store-Sales**

# Introduction

Forecast sales using store, promotion, and competitor data

---

Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.

In the program, we use a variety of models including linear models, tree models, ensemble models, specifically linear regression, decision tree, extra trees, random forest, gradient boosting and the best performing xgboost.

Thus, we choose the xgboost as our final model to predict the sales of test data.

In order to get better performance, we not only used the original train_data and store_data features, we also analyzed that time is important for sales, so we added day of year, day of month, month, year and etc. At the same time, we convert the promo Interval to whether the current time is in the promo interval which indicates the sales might be higher than usual.

---

## How to run the code

The working directory needs to be in the same directory as *[StoreSalesMain.py](storesalesmain.py)*  (main program file)

You can get help by typing 

```bash
python3 StoreSalesMain.py -h

usage: StoreSalesMain.py [-h] [--noPlot] [--model MODEL] [--predict] [--nfolds NFOLDS]

optional arguments:
  -h, --help       show this help message and exit
  --noPlot         not to generate some plots to analyse data
  --model MODEL    linear or decisionTree or extraTrees, gradientBoosting or randomForest or xgboost
  --predict        predict the test data and generate submission.csv by generated xgboost model directly
  --nfolds NFOLDS  Number of folds. Must be at least 2 default:10
```

You can run and get the results, plots and submission.csv without training the model by typing:

**Note**: You need to download the Xgboost.pkl file from google drive: [https://drive.google.com/file/d/1lPzG-XooVw5cA4QOp6GXJzChzzBhKW29/view?usp=sharing](https://drive.google.com/file/d/1lPzG-XooVw5cA4QOp6GXJzChzzBhKW29/view?usp=sharing) and place it in the same directory as [StoreSalesMain.py](storesalesmain.py/)

```bash
python3 StoreSalesMain.py --predict
```

If you **don't want** to obtain the Data analysis graphs

```bash
python3 StoreSalesMain.py --noPlot
```

Train the *linear regression model* and get the cross validation score

```bash
python3 StoreSalesMain.py --model linear --nfolds 10 
```

get the process of training *xgboost* and get the plots of predict data as well as the **submission.csv**

```bash
python3 StoreSalesMain.py --model xgboost
```

### Command to train extra models

Get the cross validation score of *decision tree* model:

```bash
python3 StoreSalesMain.py --model decisionTree --nfolds 10 
```

Get the cross validation score of *extra trees* model:

```bash
python3 StoreSalesMain.py --model extraTrees --nfolds 10 
```

You can train the gradient boosting model and get the cross validation score

```bash
python3 StoreSalesMain.py --model gradientBoosting --nfolds 10 
```

You can train the random forest model and get the cross validation score

```bash
python3 StoreSalesMain.py --model randomForest --nfolds 10 
```

## File introduction

```bash
├── GeneratePlot.py             // Generate data analysis graph
├── LossFuction.py              // Loss fuction 
├── Model.py                    // Train different models
├── StoreSalesMain.py           // Main program file
├── Xgboost.pkl                 // The pre-trained model of xgboost
├── README.md                   // Readme file
├── input                       // Data folder
│   ├── store.csv               // Data of stores
│   ├── test.csv                // Data of test
│   └── train.csv               // Data of training
├── Rossmann-Store-Sales.ipynb  // jupyter notebook that has cached results
└── submission.csv              // the predict data of test

```