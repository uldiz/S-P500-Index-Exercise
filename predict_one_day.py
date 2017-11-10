import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

# loading in data
sp = pd.read_csv("sphist.csv") 
# converting Date column to pd date type. 
sp["Date"] = pd.to_datetime(sp["Date"])
# sort the df on Date col in asc order
sp.sort("Date", ascending=True, inplace=True)
# print out the first 5 rows
print(sp.head(n=5))

sp["Close"] = (sp["Close"] - np.mean(sp["Close"])) / np.std(sp["Close"]) 
sp["Av_5day"] = sp["Close"].rolling(window=5).mean().shift(1)
sp["Av_30day"] = sp["Close"].rolling(window=30).mean().shift(1)
sp["Av_1yr"] = sp["Close"].rolling(window=365).mean().shift(1)
sp["ratio_mean_5day_1yr"] = sp["Av_5day"]/sp["Av_1yr"]
sp["sd_5day"] = sp["Close"].rolling(window=5).std().shift(1)
sp["sd_30day"] = sp["Close"].rolling(window=30).std().shift(1)
sp["sd_365day"] = sp["Close"].rolling(window=365).std().shift(1)
sp["ratio_std_5day_1yr"] = sp["sd_5day"]/sp["sd_365day"]
sp["ratio_std_5day_30day"] = sp["sd_5day"]/sp["sd_30day"]

sp = sp[sp["Date"] >= datetime(year=1951, month=1, day=3)]
sp.dropna(axis=0, inplace=True)

train = sp[sp["Date"] <= datetime(year=2013, month=1, day=3)]
test = sp[sp["Date"] == datetime(year=2013, month=1, day=4)]
features = ["Av_5day", "Av_30day", "Av_1yr", "ratio_mean_5day_1yr", "sd_5day", "sd_30day", "sd_365day", "ratio_std_5day_1yr", "ratio_std_5day_30day"]
y_train = train["Close"]
y_test = test["Close"]

lr = LinearRegression()
lr.fit(train[features], y_train)
predictions = lr.predict(test[features])

mse = np.sum((predictions - y_test)**2)/len(predictions)
print(mse)


