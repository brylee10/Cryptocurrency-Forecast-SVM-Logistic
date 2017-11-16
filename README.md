# Cryptocurrency-Forecast-SVM-Logistic
Uses SVM and logistic regression to predict cryptocurrency prices.

I used logistic regression and support vector machines (RBF and the Linear classifier) to predict if Bitcoin would increase on a given day. I used timeseries data from Crypto Compare to train each of the two classification models with data on the amount the price of the currency had changed each of the past two days (done by shifting the timeseries data). The algorithms were given if the currency did increase the day after each two-day period. The input data of opening and closing prices was randomly permuted, and the first 80% was used as training data and the last 20% was used as testing data. This process was repeated 20 times and the total correct predictions were averaged across the trials. 

Results:
SVM (RBF)- 50.5%
SVM (Lin)- 52.2%
Logistic Regression- 52.6%

The results showed slightly better than guessing performance for logistic regression and the linear SVM.
