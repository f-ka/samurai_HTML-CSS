from email.policy import default
from random import choices
from sklearn.exceptions import DataDimensionalityWarning
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
import numpy as np
import talib
import lightgbm as lgb

ticker = yf.Ticker("7203.T")

hist = ticker.history(period="max")
df_diff = hist["Close"].diff(1)
df_up, df_down = df_diff.copy(), df_diff.copy()
df_up[df_up < 0] = 0
df_down[df_down > 0] = 0
df_down = df_down * -1
df_up_sma14 = df_up.rolling(window=14, center=False).mean()
df_down_sma14 = df_down.rolling(window=14, center=False).mean()
hist["RSI"] = 100.0 * (df_up_sma14 / (df_up_sma14 + df_down_sma14))
hist["Buy"] = hist["RSI"] < 30
hist["Sell"] = hist["RSI"] > 70
hist = hist[hist["Buy"] | hist["Sell"] == True]
print(hist)
# hist["Return"] = hist.Close.diff().shift(-1)
