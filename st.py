import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

data = pd.read_csv("topix500.csv", nrows=1)
#data = pd.read_csv("sample.csv")
#data = pd.read_csv("one.csv")
print(data)

stocks = [str(s)+".T" for s in data.code]
stocks.append("^N225")
tickers = yf.Tickers(" ".join(stocks))

closes   = [] # 終値

dummy_ticker = None
for i in tickers.tickers:
    if dummy_ticker == None:
        dummy_ticker = i
    closes.append(tickers.tickers[i].history(period="1y").Close)

# 時間によって止まる？
closes = pd.DataFrame(closes).T   # DataFrame化
#closes = closes.T
closes.columns = stocks           # カラム名の設定
closes = closes.ffill()           # 欠損データの補完

print(closes)

earnings = [] # 当期純利益

dummy = tickers.tickers[dummy_ticker].financials.T["Net Income"]
dummy[:] = np.nan

for i in tickers.tickers:
    try:
        earnings.append(tickers.tickers[i].financials.T["Net Income"])
    except:
        earnings.append(dummy)       # エラー発生時はダミーを入れる

earnings = pd.DataFrame(earnings).T  # DataFrame化
earnings.columns = stocks            # カラム名の設定

print(earnings)

equity   = [] # 自己資本

dummy = tickers.tickers[dummy_ticker].balance_sheet.T["Total Stockholder Equity"]
dummy[:] = np.nan

for i in tickers.tickers:
    try:
        equity.append(tickers.tickers[i].balance_sheet.T["Total Stockholder Equity"])
    except:
        equity.append(dummy)         # エラー発生時はダミーを入れる

equity = pd.DataFrame(equity).T      # DataFrame化
equity.columns = stocks              # カラム名の設定

print(equity)

shares   = [] # 発行株数

for i in tickers.tickers:
    try:
        shares.append(tickers.tickers[i].info["sharesOutstanding"])
    except:
        shares.append(np.nan)        # エラー発生時はNAN値を入れる

shares = pd.Series(shares)           # Series化
shares.index = stocks                # インデックス名の設定

print(shares)

eps = earnings/shares.values      # EPS
roe = earnings/equity             # ROE

eps = eps.ffill()                 # 欠損データの補完
roe = roe.ffill()

eps = eps.drop(["^N225"], axis=1) # ^N225カラムは削除しておく
roe = roe.drop(["^N225"], axis=1)

print(eps)
print(roe)

closes["month"] = closes.index.month                                      # 月カラムの作成
closes["end_of_month"] = closes.month.diff().shift(-1)                    # 月末フラグカラムの作成
closes = closes[closes.end_of_month != 0]                                 # 月末のみ抽出

monthly_rt = closes.pct_change().shift(-1)                                # 月次リターンの作成(ラグあり)
#monthly_rt = closes.pct_change(12).shift(-1)                                # 年次リターンの作成(ラグあり)
monthly_rt = monthly_rt.sub(monthly_rt["^N225"], axis=0)                  # マーケットリターン控除

closes = closes[closes.index > datetime.datetime(2018, 4, 1)]             # 2017年4月以降
monthly_rt = monthly_rt[monthly_rt.index > datetime.datetime(2018, 4, 1)]

closes = closes.drop(["^N225", "month", "end_of_month"], axis=1)          # 不要なカラムを削除
monthly_rt = monthly_rt.drop(["^N225", "month", "end_of_month"], axis=1)

print(closes)
print(monthly_rt)

eps_df = pd.DataFrame(index=monthly_rt.index, columns=monthly_rt.columns) # 月次リターンと同次元のDF作成
roe_df = pd.DataFrame(index=monthly_rt.index, columns=monthly_rt.columns)

for i in range(len(eps_df)):                                              # 各行への代入
    eps_df.iloc[i] = eps[eps.index < eps_df.index[i]].iloc[-1]

for i in range(len(roe_df)):
    roe_df.iloc[i] = roe[roe.index < roe_df.index[i]].iloc[-1]

per_df = closes/eps_df                                                    # PERデータフレームの作成

print(per_df)
print(roe_df)

stack_monthly_rt = monthly_rt.stack()                                  # 1次元にスタック
stack_per_df = per_df.stack()
stack_roe_df = roe_df.stack()

df = pd.concat([stack_monthly_rt, stack_per_df, stack_roe_df], axis=1) # 結合
df.columns = ["rt", "per", "roe"]                                      # カラム名の設定

df["rt"][df.rt > 1.0] = np.nan                                         # 異常値の除去

print(df)

value_df = df[(df.per < 10) & (df.roe > 0.1)]       # 割安でクオリティが高い銘柄を抽出

plt.hist(value_df["rt"])                            # ヒストグラムの描画
plt.show()

balance = value_df.groupby(level=0).mean().cumsum() # 累積リターンを作成

plt.clf()
plt.plot(balance["rt"])                             # バランスカーブの描画
plt.show()
