import yfinance as yf
import matplotlib.pyplot as plt

ticker = yf.Ticker("9984.T")

# get historical market data
hist = ticker.history(period="2y")
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

hist = hist[hist["gc"] != 0 or hist["dc"] != 0]

print(hist)

"""
periods = [5, 25, 75]
cols = []
for period in periods:
    col = "{} windows simple moving average".format(period)
    hist[col] = hist.Close.rolling(period, min_periods=1).mean()
    cols.append(col)
for col in cols:
    plt.plot(hist[col], label=col)
"""

plt.subplots(figsize=(15, 5))
# plt.plot(hist.Close)
plt.plot(sma_1, label="Moving Average {} days".format(ma_1))
plt.plot(sma_2, label="Moving Average {} days".format(ma_2))
# plt.plot(sma_3, label="Moving Average {} days".format(ma_3))
plt.scatter(gc.index, gc, label="Golden Cross", s=50, c="red", alpha=0.7)
plt.scatter(dc.index, dc, label="Dead Cross", s=50, c="black", alpha=0.7)
plt.grid(True)
plt.legend()
plt.show()

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
