from email.policy import default
from random import choices
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
import numpy as np
import talib
import lightgbm as lgb

ticker = yf.Ticker("9984.T")

# get historical market data
hist = ticker.history(period="max")
# valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
# valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

ma_1 = 5
ma_2 = 25
# ma_3 = 75

hist["sma_1"] = hist.Close.rolling(window=ma_1, min_periods=1).mean()
hist["sma_2"] = hist.Close.rolling(window=ma_2, min_periods=1).mean()
sma_1 = hist.Close.rolling(window=ma_1, min_periods=1).mean()
sma_2 = hist.Close.rolling(window=ma_2, min_periods=1).mean()
# sma_3 = hist.Close.rolling(window=ma_3, min_periods=1).mean()

diff = sma_1 - sma_2
hist["gc"] = (diff.shift(1) < 0) & (diff > 0)
hist["dc"] = (diff.shift(1) > 0) & (diff < 0)
# gc = sma_1[(diff.shift(1) < 0) & (diff > 0)]
# dc = sma_1[(diff.shift(1) > 0) & (diff < 0)]

hist = hist[hist["gc"] | hist["dc"] == True]
hist["Return"] = hist.Close.diff().shift(-1)
# hist = hist[hist["dc"] == True]
conditions = [hist["Return"] > 0]
choices = [1]
hist["Result"] = np.select(conditions, choices, default=0)
print(hist)
# Total = hist["Return"].sum()
# print("Total: ", Total)
# train = hist["2000-01-04":"2015-12-31"]
# print(train)
# test = hist["2016-01-01":]
# print(test)


def calc_features(hist):
    open = hist["Open"]
    high = hist["High"]
    low = hist["Low"]
    close = hist["Close"]
    volume = hist["Volume"]

    orig_columns = hist.columns

    hilo = (hist["High"] + hist["Low"]) / 2
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
# print(hist)

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
        #     'MINUS_DI',
        #     'MINUS_DM',
        "MOM",
        #     'PLUS_DI',
        #     'PLUS_DM',
        "RSI",
        "STOCH_slowk",
        "STOCH_slowd",
        "STOCHF_fastk",
        #     'STOCHRSI_fastd',
        "ULTOSC",
        "WILLR",
        #     'ADOSC',
        #     'NATR',
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
    ]
)

# print(features)

hist = hist.dropna()
model = lgb.LGBMRegressor(n_jobs=-1, random_state=1)
model.fit(hist[features], hist["Result"])

cv_indicies = list(KFold().split(hist))


def my_cross_val_predict(estimator, X, y=None, cv=None):
    y_pred = y.copy()
    y_pred[:] = np.nan
    for train_idx, val_idx in cv:
        estimator.fit(X[train_idx], y[train_idx])
        y_pred[val_idx] = estimator.predict(X[val_idx])
    return y_pred


hist["pred_Return"] = my_cross_val_predict(
    model, hist[features].values, hist["Result"].values, cv=cv_indicies
)
hist = hist.dropna()

print("毎時刻、pred_Returnがプラスのときだけトレードした場合の累積リターン")
hist[hist["pred_Return"] > 0]["Result"].cumsum().plot(label="買い")
plt.title("累積リターン")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.show()

hist = hist[hist["pred_Return"] > 0]
print(hist)
Total = hist["Return"].sum()
print("Total: ", Total)

# x_train = train[["Open", "High", "Low", "Close", "Volume", "sma_1", "sma_2"]]
# y_train = train["Result"]

# x_test = test[["Open", "High", "Low", "Close", "Volume", "sma_1", "sma_2"]]
# y_test = test["Result"]


# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# model.fit(x_train, y_train)
# pred = model.predict(x_test)

# print("result: ", model.score(x_test, y_test))
# print(classification_report(y_test, pred))

"""
# periods = [5, 25, 75]
# cols = []
# for period in periods:
#     col = "{} windows simple moving average".format(period)
#     hist[col] = hist.Close.rolling(period, min_periods=1).mean()
#     cols.append(col)
# for col in cols:
#     plt.plot(hist[col], label=col)
"""

# plt.subplots(figsize=(15, 5))
# plt.plot(hist.Close)
# plt.plot(sma_1, label="Moving Average {} days".format(ma_1))
# plt.plot(sma_2, label="Moving Average {} days".format(ma_2))
# # plt.plot(sma_3, label="Moving Average {} days".format(ma_3))
# plt.scatter(gc.index, gc, label="Golden Cross", s=50, c="red", alpha=0.7)
# plt.scatter(dc.index, dc, label="Dead Cross", s=50, c="black", alpha=0.7)
# plt.grid(True)
# plt.legend()
# plt.show()


"""
# get stock info
ticker.info

# show actions (dividends, splits)
ticker.actions

# show dividends
ticker.dividends

# show splits
ticker.splits

# show financials
ticker.financials
ticker.quarterly_financials

# show major holders
ticker.major_holders

# show institutional holders
ticker.institutional_holders

# show balance sheet
ticker.balance_sheet
ticker.quarterly_balance_sheet

# show cashflow
ticker.cashflow
ticker.quarterly_cashflow

# show earnings
ticker.earnings
ticker.quarterly_earnings

# show sustainability
ticker.sustainability

# show analysts recommendations
ticker.recommendations

# show next event (earnings, etc)
ticker.calendar

# show ISIN code - *experimental*
# ISIN = International Securities Identification Number
ticker.isin

# show options expirations
ticker.options

# show news
ticker.news

# get option chain for specific expiration
opt = ticker.option_chain('YYYY-MM-DD')
# data available via: opt.calls, opt.puts
"""
