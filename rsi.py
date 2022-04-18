import pandas as pd
import yfinance as yf

data = pd.read_csv("topix.csv")
codes = [str(s) + ".T" for s in data.code]
hist = dict()
for code in codes:
    ticker = yf.Ticker(code)
    hist[code] = ticker.history(period="max")

    df_diff = hist[code]["Close"].diff(1)
    df_up, df_down = df_diff.copy(), df_diff.copy()
    df_up[df_up < 0] = 0
    df_down[df_down > 0] = 0
    df_down = df_down * -1
    df_up_sma14 = df_up.rolling(window=14, center=False).mean()
    df_down_sma14 = df_down.rolling(window=14, center=False).mean()
    hist[code]["RSI"] = 100.0 * (df_up_sma14 / (df_up_sma14 + df_down_sma14))

    hist[code]["Under"] = (hist[code]["RSI"].shift() > 30) & (hist[code]["RSI"] < 30)
    hist[code]["Over"] = (hist[code]["RSI"].shift() < 70) & (hist[code]["RSI"] > 70)
    hist[code] = hist[code][hist[code]["Under"] | hist[code]["Over"] == True]
    hist[code]["Buy"] = (hist[code]["Under"] == True) & (
        hist[code]["Under"].shift(1) == False
    )
    hist[code]["Sell"] = (hist[code]["Over"] == True) & (
        hist[code]["Over"].shift(1) == False
    )
    hist[code] = hist[code][hist[code]["Buy"] | hist[code]["Sell"] == True]
    hist[code]["Return"] = hist[code].Close.diff().shift(-1)
    hist[code] = hist[code][hist[code]["Buy"] == True]

hist = pd.concat(hist)
print(hist)
hist.to_csv("hist_rsi.csv")
