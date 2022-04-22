import math

from crypto_data_fetcher.gmo import GmoFetcher
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
import seaborn as sns
import talib

from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from IPython.core.display import display

memory = joblib.Memory("/tmp/gmo_fetcher_cache", verbose=0)
fetcher = GmoFetcher(memory=memory)

# GMOコインのBTC/JPYレバレッジ取引 ( https://api.coin.z.com/data/trades/BTC_JPY/ )を取得
# 初回ダウンロードは時間がかかる
df = fetcher.fetch_ohlcv(
    market="BTC_JPY",  # 市場のシンボルを指定
    interval_sec=15 * 60,  # 足の間隔を秒単位で指定。この場合は15分足
)

# 実験に使うデータ期間を限定する
df = df[df.index < pd.to_datetime("2021-04-01 00:00:00Z")]

display(df)
df.to_pickle("df_ohlcv.pkl")
