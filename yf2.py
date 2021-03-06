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
            share.PERIOD_TYPE_YEAR, 20, share.FREQUENCY_TYPE_DAY, 1
        )
    except YahooFinanceError as e:
        print(e.message)
        sys.exit(1)

    hist[code] = pd.DataFrame(symbol_data)
    hist[code]["datetime"] = pd.to_datetime(hist[code].timestamp, unit="ms")

    ma_1 = 5
    ma_2 = 25

    hist[code]["sma_1"] = hist[code].close.rolling(window=ma_1, min_periods=1).mean()
    hist[code]["sma_2"] = hist[code].close.rolling(window=ma_2, min_periods=1).mean()
    sma_1 = hist[code].close.rolling(window=ma_1, min_periods=1).mean()
    sma_2 = hist[code].close.rolling(window=ma_2, min_periods=1).mean()

    diff = sma_1 - sma_2
    hist[code]["gc"] = (diff.shift(1) < 0) & (diff > 0)
    hist[code]["dc"] = (diff.shift(1) > 0) & (diff < 0)

    hist[code] = hist[code][hist[code]["gc"] | hist[code]["dc"] == True]
    hist[code]["Return"] = hist[code].close.diff().shift(-1)
    hist[code] = hist[code][hist[code]["gc"] == True]

hist = pd.concat(hist)
print(hist)
r1 = hist["Return"].sum()


def calc_features(hist):
    open = hist["open"]
    high = hist["high"]
    low = hist["low"]
    close = hist["close"]
    volume = hist["volume"]

    hilo = (hist["high"] + hist["low"]) / 2
    (
        hist["BBANDS_upperband"],
        hist["BBANDS_middleband"],
        hist["BBANDS_lowerband"],
    ) = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    hist["BBANDS_upperband"] -= hilo
    hist["BBANDS_middleband"] -= hilo
    hist["BBANDS_lowerband"] -= hilo
    hist["DEMA"] = talib.DEMA(close, timeperiod=30) - hilo
    hist["EMA"] = talib.EMA(close, timeperiod=30) - hilo
    hist["HT_TRENDLINE"] = talib.HT_TRENDLINE(close) - hilo
    hist["KAMA"] = talib.KAMA(close, timeperiod=30) - hilo
    hist["MA"] = talib.MA(close, timeperiod=30, matype=0) - hilo
    hist["MIDPOINT"] = talib.MIDPOINT(close, timeperiod=14) - hilo
    hist["SMA"] = talib.SMA(close, timeperiod=30) - hilo
    hist["T3"] = talib.T3(close, timeperiod=5, vfactor=0) - hilo
    hist["TEMA"] = talib.TEMA(close, timeperiod=30) - hilo
    hist["TRIMA"] = talib.TRIMA(close, timeperiod=30) - hilo
    hist["WMA"] = talib.WMA(close, timeperiod=30) - hilo

    hist["ADX"] = talib.ADX(high, low, close, timeperiod=14)
    hist["ADXR"] = talib.ADXR(high, low, close, timeperiod=14)
    hist["APO"] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    hist["AROON_aroondown"], hist["AROON_aroonup"] = talib.AROON(
        high, low, timeperiod=14
    )
    hist["AROONOSC"] = talib.AROONOSC(high, low, timeperiod=14)
    hist["BOP"] = talib.BOP(open, high, low, close)
    hist["CCI"] = talib.CCI(high, low, close, timeperiod=14)
    hist["DX"] = talib.DX(high, low, close, timeperiod=14)
    hist["MACD_macd"], hist["MACD_macdsignal"], hist["MACD_macdhist"] = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    hist["MFI"] = talib.MFI(high, low, close, volume, timeperiod=14)
    hist["MINUS_DI"] = talib.MINUS_DI(high, low, close, timeperiod=14)
    hist["MINUS_DM"] = talib.MINUS_DM(high, low, timeperiod=14)
    hist["MOM"] = talib.MOM(close, timeperiod=10)
    hist["PLUS_DI"] = talib.PLUS_DI(high, low, close, timeperiod=14)
    hist["PLUS_DM"] = talib.PLUS_DM(high, low, timeperiod=14)
    hist["RSI"] = talib.RSI(close, timeperiod=14)
    hist["STOCH_slowk"], hist["STOCH_slowd"] = talib.STOCH(
        high,
        low,
        close,
        fastk_period=5,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0,
    )
    hist["STOCHF_fastk"], hist["STOCHF_fastd"] = talib.STOCHF(
        high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0
    )
    hist["STOCHRSI_fastk"], hist["STOCHRSI_fastd"] = talib.STOCHRSI(
        close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0
    )
    hist["TRIX"] = talib.TRIX(close, timeperiod=30)
    hist["ULTOSC"] = talib.ULTOSC(
        high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28
    )
    hist["WILLR"] = talib.WILLR(high, low, close, timeperiod=14)
    hist["AD"] = talib.AD(high, low, close, volume)
    hist["ADOSC"] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    hist["OBV"] = talib.OBV(close, volume)

    hist["ATR"] = talib.ATR(high, low, close, timeperiod=14)
    hist["NATR"] = talib.NATR(high, low, close, timeperiod=14)
    hist["TRANGE"] = talib.TRANGE(high, low, close)

    hist["HT_DCPERIOD"] = talib.HT_DCPERIOD(close)
    hist["HT_DCPHASE"] = talib.HT_DCPHASE(close)
    hist["HT_PHASOR_inphase"], hist["HT_PHASOR_quadrature"] = talib.HT_PHASOR(close)
    hist["HT_SINE_sine"], hist["HT_SINE_leadsine"] = talib.HT_SINE(close)
    hist["HT_TRENDMODE"] = talib.HT_TRENDMODE(close)

    hist["BETA"] = talib.BETA(high, low, timeperiod=5)
    hist["CORREL"] = talib.CORREL(high, low, timeperiod=30)
    hist["LINEARREG"] = talib.LINEARREG(close, timeperiod=14) - close
    hist["LINEARREG_ANGLE"] = talib.LINEARREG_ANGLE(close, timeperiod=14)
    hist["LINEARREG_INTERCEPT"] = (
        talib.LINEARREG_INTERCEPT(close, timeperiod=14) - close
    )
    hist["LINEARREG_SLOPE"] = talib.LINEARREG_SLOPE(close, timeperiod=14)
    hist["STDDEV"] = talib.STDDEV(close, timeperiod=5, nbdev=1)

    return hist


hist = hist.dropna()
hist = calc_features(hist)
print(hist)
features = sorted(
    [
        "ADX",
        "ADXR",
        "APO",
        "AROON_aroondown",
        "AROON_aroonup",
        "AROONOSC",
        "CCI",
        "DX",
        "MACD_macd",
        "MACD_macdsignal",
        "MACD_macdhist",
        "MFI",
        "MINUS_DI",
        "MINUS_DM",
        "MOM",
        "PLUS_DI",
        "PLUS_DM",
        "RSI",
        "STOCH_slowk",
        "STOCH_slowd",
        "STOCHF_fastk",
        "STOCHRSI_fastd",
        "ULTOSC",
        "WILLR",
        "ADOSC",
        "NATR",
        "HT_DCPERIOD",
        "HT_DCPHASE",
        "HT_PHASOR_inphase",
        "HT_PHASOR_quadrature",
        "HT_TRENDMODE",
        "BETA",
        "LINEARREG",
        "LINEARREG_ANGLE",
        "LINEARREG_INTERCEPT",
        "LINEARREG_SLOPE",
        "STDDEV",
        "BBANDS_upperband",
        "BBANDS_middleband",
        "BBANDS_lowerband",
        "DEMA",
        "EMA",
        "HT_TRENDLINE",
        "KAMA",
        "MA",
        "MIDPOINT",
        "T3",
        "TEMA",
        "TRIMA",
        "WMA",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sma_1",
        "sma_2",
    ]
)
hist = hist.dropna()
# model = lgb.LGBMRegressor(n_jobs=-1, random_state=1)
model = RandomForestRegressor(n_estimators=100, max_depth=5, n_jobs=-1, random_state=1)
model.fit(hist[features], hist["Return"])
df = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
df = df.sort_values("importance", ascending=False)
df = df.reset_index()
print(df)

cv_indicies = list(KFold().split(hist))


def my_cross_val_predict(estimator, X, y=None, cv=None):
    y_pred = y.copy()
    y_pred[:] = np.nan
    for train_idx, val_idx in cv:
        estimator.fit(X[train_idx], y[train_idx])
        y_pred[val_idx] = estimator.predict(X[val_idx])
    return y_pred


hist["pred_Return"] = my_cross_val_predict(
    model, hist[features].values, hist["Return"].values, cv=cv_indicies
)
hist = hist.dropna()
hist = hist[hist["pred_Return"] > 0]
r2 = hist["Return"].sum()

print("????????????: ", r1)
print("??????????????????: ", r2)
