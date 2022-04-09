import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import pickle

df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)


X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]   ## 165-200 that is the end of data ,- slicing means that we're refering the end of the array  , arr = np.array([1, 2, 3, 4, 5, 6, 7])
                                # therefore print(arr[-3:-1])  >> [5 6] 
X = X[:-forecast_out]           ## 0-165

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test =  model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

## pickling after training 

## With pickle, you can save any Python object, like our classifier. After defining, training, and testing your classifier, add:
with open('linearregression.pickle','wb') as f: ## f is a temporary variable    ## wb is not a pickle parameter, but a parameter for the open function. It means: open for writing and open in binary mode
    pickle.dump(clf, f)         ## we dump the trained classifier clf to f 

## Now, run the script again, and boom, you should have linearregression.pickle which is the serialized data for the classifier. 
 # Now, all you need to do to use the classifier is load in the pickle, save it to clf, and use just like normal. For example:
pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)  ## clf is loaded with the trained data 

## after pickle data is saved we can comment out lines 32-39  since we will then just load data

confidence = clf.score(X_test, y_test)
print(confidence)


forecast_set = clf.predict(X_lately)  ## pass either a single value or an array to predict values on


print(forecast_set, confidence, forecast_out)  ## forecast_set= forecast confidence as a whole on days ahead data,confidence = prediction model values on the forecast_out days , forecast_out = Number of days we're predicting on 


import datetime
import matplotlib.pyplot as plt
from matplotlib import style


style.use('ggplot')


df['Forecast'] = np.nan


## iloc =- Purely integer-location based indexing for selection by position.
 # loc = Access a group of rows and columns by label(s) or a boolean array.

last_date = df.iloc[-1].name  ## .name here refers to The name of a Series within a DataFrame is its column name.
## df["Even Numbers"].name >> 'Even Numbers'

last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day


## code below is to have dates in the x-axis
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

## So here all we're doing is iterating through the forecast set, taking each forecast and day, and then setting those values in the dataframe 
 # (making the future "features" NaNs) basically everything as NaN in the dataframe . The last line's code just simply takes all of the first columns, setting them to NaNs, and then the final column
 # is whatever i is (the forecast in this case). I have chosen to do this one-liner for loop like this so that, if we decide to change up the dataframe and features,
 # the code can still work

print(df.head)
print(df.tail)

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


