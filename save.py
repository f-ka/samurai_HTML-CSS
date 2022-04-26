from email.policy import default
from random import choices
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
import numpy as np
import talib
import warnings

warnings.filterwarnings("ignore")
import lightgbm as lgb
import random
from IPython.core.display import display
import sys
import pandas as pd
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError

data = pd.read_csv("topix500.csv")
codes = [str(s) + ".T" for s in data.code]
hist = dict()
for code in codes:
    my_share = share.Share(code)
    symbol_data = None

    try:
        symbol_data = my_share.get_historical(
            share.PERIOD_TYPE_YEAR, 5, share.FREQUENCY_TYPE_DAY, 1
        )
    except YahooFinanceError as e:
        print(e.message)
        sys.exit(1)

    hist[code] = pd.DataFrame(symbol_data)
    hist[code]["datetime"] = pd.to_datetime(hist[code].timestamp, unit="ms")
    print(hist)
    hist[code].to_csv("./hist/" + code + ".csv")
