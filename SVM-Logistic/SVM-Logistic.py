### REFERENCES:
# Time Series code from: http://www.quantatrisk.com/2017/03/20/download-crypto-currency-time-series-portfolio-python/
# Time Series Shifting: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
# SVM Code: https://github.com/dataventures/spring2017/blob/master/3/classification_preview.ipynb
# Logistic Regression: http://www.data-mania.com/blog/logistic-regression-example-in-python/

# Author: Bryan Lee
# Date: 10/19/17

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot as plt
from datetime import datetime
import json
from bs4 import BeautifulSoup
import requests
from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression

## Parameters
shiftVal = 3;
trainProp = 8/10
numIterations = 20

def timestamp2date(timestamp):
    # function converts a Unix timestamp into Gregorian date
    return datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d')


def date2timestamp(date):
    # function converts Gregorian date in a given format to timestamp
    return datetime.strptime(date.today(), '%Y-%m-%d').timestamp()


def fetchCryptoOHLC(fsym, tsym):
    # function fetches a crypto price-series for fsym/tsym and stores
    # it in pandas DataFrame

    cols = ['date', 'timestamp', 'open', 'high', 'low', 'close']
    lst = ['time', 'open', 'high', 'low', 'close']

    timestamp_today = datetime.today().timestamp()
    curr_timestamp = timestamp_today

    for j in range(2):
        df = pd.DataFrame(columns=cols)
        url = "https://min-api.cryptocompare.com/data/histoday?fsym=" + fsym + "&tsym=" + tsym + "&toTs=" + str(int(curr_timestamp)) + "&limit=2000"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        dic = json.loads(soup.prettify())
        for i in range(1, 2001):
            tmp = []
            for e in enumerate(lst):
                x = e[0]
                y = dic['Data'][i][e[1]]
                if (x == 0):
                    tmp.append(str(timestamp2date(y)))
                tmp.append(y)
            if (np.sum(tmp[-4::]) > 0):
                df.loc[len(df)] = np.array(tmp)
        df.index = pd.to_datetime(df.date)
        df.drop('date', axis=1, inplace=True)
        curr_timestamp = int(df.ix[0][0])
        if (j == 0):
            df0 = df.copy()
        else:
            data = pd.concat([df, df0], axis=0)


    return data

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

fsym = "BTC"
tsym = "USD"
data = fetchCryptoOHLC(fsym, tsym)

# add new columns

open = np.array(list(map(float, data['open'].values)))
close = np.array(list(map(float, data['close'].values)))

data['Change'] = pd.Series(close - open, index=data.index)
dataCopy = DataFrame()
dataCopy['Change'] = data['Change']
dataCopy['timestamp'] = data['timestamp']
shiftedData = series_to_supervised(dataCopy.values, shiftVal)
#print(temp)

mlData = DataFrame()
mlData[1] = shiftedData['var1(t)']
mlData[2] = shiftedData['var1(t-1)']
mlData[3] = shiftedData['var1(t-2)']
mlData[4] = shiftedData['var1(t-3)']

shiftedColumns = mlData[[2, 3]].copy()
shiftedColumns = np.array(shiftedColumns)

#mlData = mlData.as_matrix

# determines if actually appreciated or not
result = data['Change']
result = np.array(result)
result[result < 0] = 0;
result[result > 0] = 1;

correctCntSVM = [0] * 2
correctCntRegression = 0
for i in range(numIterations):
    X_total = np.random.permutation(shiftedColumns)
    y_total = np.random.permutation(result[shiftVal: ])
    #X_total = shiftedColumns
    #y_total = result[shiftVal: ]

    X = X_total[ : round(len(X_total)*trainProp)]
    y = y_total[ : round(len(X_total)*trainProp)]


    ## Support Vector Machine
    C = 1.0  # SVM regularization parameter
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
    lin_svc = svm.LinearSVC(C=C).fit(X, y)

    total = len(X_total) - round(len(X_total) * trainProp) + 1
    for i in range(round(len(X_total) * trainProp) + 1, len(X_total)):
        if (rbf_svc.predict([X_total[i]]) == y_total[i]):
            correctCntSVM[0] += 1

        if (lin_svc.predict([X_total[i]]) == y_total[i]):
            correctCntSVM[1] += 1

    ## Logistic Regression
    logistic = LogisticRegression()
    logistic.fit(X, y)

    for i in range(round(len(X_total) * trainProp) + 1, len(X_total)):
        if (logistic.predict([X_total[i]]) == y_total[i]):
            correctCntRegression += 1


# Print Results
SVM_Names = ['RBF', 'Lin']
print ('Iterations: %s' % (numIterations))
print ('Correct Predicted Percentage by SVM: ')
for i in range(2):
    print ('%s: %s' % (SVM_Names[i], correctCntSVM[i]/(total*numIterations)))

print ('\nCorrect Predicted Percentage by Logistic Regression: \n%s' % (correctCntRegression/(total*numIterations)))
