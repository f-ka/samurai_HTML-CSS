from django.shortcuts import render, redirect
from .forms import PostForm
from django.utils import timezone
from sklearn.externals import joblib
from django.http import HttpResponse
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
import pickle
import glob
import os


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


def stocks(request):
    if request.method == "POST":
        files = glob.glob("hist_month/*.csv")
        print(files)
        hist = dict()
        for file in files:
            code = os.path.splitext(os.path.basename(file))[0]
            hist[code] = pd.read_csv(file)

            hist[code] = calc_features(hist[code])

            hist[code] = hist[code].tail(1)

            print(code)

            hist[code]["code"] = code

        hist = pd.concat(hist)

        hist = hist.dropna()

        features = sorted(
            [
                "ADOSC",
                "ADX",
                "ADXR",
                "APO",
                "AROON_aroondown",
                "AROON_aroonup",
                "AROONOSC",
                "BBANDS_lowerband",
                "BBANDS_middleband",
                "BBANDS_upperband",
                "BETA",
                "CCI",
                "DEMA",
                "DX",
                "EMA",
                "HT_DCPERIOD",
                "HT_DCPHASE",
                "HT_PHASOR_inphase",
                "HT_PHASOR_quadrature",
                "HT_TRENDLINE",
                "HT_TRENDMODE",
                "KAMA",
                "LINEARREG",
                "LINEARREG_ANGLE",
                "LINEARREG_INTERCEPT",
                "LINEARREG_SLOPE",
                "MA",
                "MACD_macd",
                "MACD_macdhist",
                "MACD_macdsignal",
                "MFI",
                "MIDPOINT",
                "MINUS_DI",
                "MINUS_DM",
                "MOM",
                "NATR",
                "PLUS_DI",
                "PLUS_DM",
                "RSI",
                "STDDEV",
                "STOCH_slowd",
                "STOCH_slowk",
                "STOCHF_fastk",
                "STOCHRSI_fastd",
                "T3",
                "TEMA",
                "TRIMA",
                "ULTOSC",
                "WILLR",
                "WMA",
            ]
        )

        # hist = hist.dropna()

        loaded_model = pickle.load(open("finalized_model.sav", "rb"))
        # loaded_model = pickle.load(open("finalized_model_rsi.sav", "rb"))

        hist["pred_Return"] = loaded_model.predict(hist[features])

        hist = hist[hist["pred_Return"] > 500]
        print(hist)
        hist = hist.reset_index()
        return render(request, "app/result.html", {"code": hist.code})
    elif request.method == "GET":
        return render(request, "app/index.html", {})


def post_new(request):
    form = PostForm()
    return render(request, "app/post_edit.html", {"form": form})
