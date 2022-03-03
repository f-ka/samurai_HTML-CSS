from flask import Flask, render_template, request, flash, send_file, make_response
from soupsieve import select
from wtforms import Form, FloatField, SubmitField, validators, ValidationError, SelectField, IntegerField
import numpy as np
from sklearn.externals import joblib
import io
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask_cors import CORS
import json
import datetime
import pandas as pd
import yfinance as yf
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg

# 学習済みモデルを読み込み利用します
def predict(parameters):
    # ニューラルネットワークのモデルを読み込み
    model = joblib.load('nn.pkl')
    params = parameters.reshape(1,-1)
    pred = model.predict(params)
    return pred

def t_predict(t_parameters):
    # ニューラルネットワークのモデルを読み込み
    t_model = joblib.load('Titanic_nn.pkl')
    t_params = t_parameters.reshape(1,-1)
    print(t_params)
    t_pred = t_model.predict(t_params)
    return t_pred

# ラベルからIrisの名前を取得します
def getName(label):
    print(label)
    if label == 0:
        return "Iris Setosa"
    elif label == 1: 
        return "Iris Versicolor"
    elif label == 2: 
        return "Iris Virginica"
    else: 
        return "Error"

def t_getName(label):
    print(label)
    if label == 0:
        return "0"
    elif label == 1: 
        return "1"
    else: 
        return "Error"

app = Flask(__name__)
#CORS(app)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'zJe09C5c3tMf5FnNL09C5d6SAzZoY'

# 公式サイト
# http://wtforms.simplecodes.com/docs/0.6/fields.html
# Flaskとwtformsを使い、index.html側で表示させるフォームを構築します。
class IrisForm(Form):
    SepalLength = FloatField("Sepal Length(cm)（蕚の長さ）",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=10)])

    SepalWidth  = FloatField("Sepal Width(cm)（蕚の幅）",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=10)])

    PetalLength = FloatField("Petal length(cm)（花弁の長さ）",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=10)])

    PetalWidth  = FloatField("petal Width(cm)（花弁の幅）",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=10)])

    # html側で表示するsubmitボタンの表示
    submit = SubmitField("判定")

class t_Form(Form):
    Pclass = SelectField("チケットクラス", choices=[1,2,3])

    Sex  = SelectField("性別", choices=[0,1])

    SibSp = SelectField("同乗している兄弟/配偶者の数", choices=[0,1,2,3,4,5,8])

    Parch  = SelectField("同乗している親/子供の数", choices=[0,1,2,3,4,5,6])

    # html側で表示するsubmitボタンの表示
    submit = SubmitField("判定")

class st_Form(Form):
    StockName = IntegerField("銘柄数",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=1, max=10)])

    Years  = IntegerField("取得年数",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=1, max=10)])
    Since  = IntegerField("",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=2018, max=2022)])
    # html側で表示するsubmitボタンの表示
    submit = SubmitField("実行")

@app.route('/', methods = ['GET', 'POST'])
def predicts():
    form = IrisForm(request.form)
    if request.method == 'POST':
        if form.validate() == False:
            flash("全て入力する必要があります。")
            return render_template('index.html', form=form)
        else:            
            SepalLength = float(request.form["SepalLength"])            
            SepalWidth  = float(request.form["SepalWidth"])            
            PetalLength = float(request.form["PetalLength"])            
            PetalWidth  = float(request.form["PetalWidth"])

            x = np.array([SepalLength, SepalWidth, PetalLength, PetalWidth])
            pred = predict(x)
            irisName = getName(pred)

            return render_template('result.html', irisName=irisName)
    elif request.method == 'GET':

        return render_template('index.html', form=form)

@app.route('/Titanic', methods = ['GET', 'POST'])
def t_predicts():
    form = t_Form(request.form)
    if request.method == 'POST':
        if form.validate() == False:
            flash("全て入力する必要があります。")
            return render_template('Titanic_index.html', form=form)
        else:            
            Pclass = float(request.form["Pclass"])        
            SibSp = float(request.form["SibSp"])            
            Parch  = float(request.form["Parch"])
            Male  = float(request.form["Sex"])
            Female  = 1 - float(request.form["Sex"])

            x = np.array([Pclass, SibSp, Parch, Female, Male])
            t_pred = t_predict(x)
            t_Name = t_getName(t_pred)

            return render_template('Titanic_result.html', t_Name=t_Name)
    elif request.method == 'GET':

        return render_template('Titanic_index.html', form=form)

@app.route('/st', methods = ['GET', 'POST'])
def st():
    form = st_Form(request.form)
    if request.method == 'POST':
        if form.validate() == False:
            flash("全て入力する必要があります。")
            return render_template('st_index.html', form=form)
        else:            
            StockName = int(request.form["StockName"])        
            Years = int(request.form["Years"])
            Since = int(request.form["Since"]) + 1
            data = pd.read_csv("topix500.csv", nrows=StockName)
            print(data)
            stocks = [str(s)+".T" for s in data.code]
            stocks.append("^N225")
            tickers = yf.Tickers(" ".join(stocks))
            closes   = [] # 終値
            dummy_ticker = None
            for i in tickers.tickers:
                if dummy_ticker == None:
                    dummy_ticker = i
                closes.append(tickers.tickers[i].history(period=f"{Years}y").Close)
            closes = pd.DataFrame(closes).T   # DataFrame化
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

            closes = closes[closes.index > datetime.datetime(Since, 4, 1)]             # 2017年4月以降
            monthly_rt = monthly_rt[monthly_rt.index > datetime.datetime(Since, 4, 1)]

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
            return render_template('st_result.html', form=form)
    elif request.method == 'GET':
        return render_template('st_index.html', form=form) 

@app.route('/Chart',methods=["GET","POST"])
def chart():
    ticker = yf.Ticker("9984.T")
    hist = ticker.history(period="2y")
    ma_1 = 5
    ma_2 = 25
    sma_1 = hist.Close.rolling(window=ma_1, min_periods=1).mean()
    sma_2 = hist.Close.rolling(window=ma_2, min_periods=1).mean()
    diff = sma_1 - sma_2
    gc = sma_1[(diff.shift(1) < 0) & (diff > 0)]
    dc = sma_1[(diff.shift(1) > 0) & (diff < 0)]
    fig = plt.figure(figsize=(15, 5))
    plt.plot(sma_1, label="Moving Average {} days".format(ma_1))
    plt.plot(sma_2, label="Moving Average {} days".format(ma_2))
    plt.scatter(gc.index, gc, label="Golden Cross", s=50, c="red", alpha=0.7)
    plt.scatter(dc.index, dc, label="Dead Cross", s=50, c="black", alpha=0.7)
    plt.grid(True)
    plt.legend()
    canvas = FigureCanvasAgg(fig)
    png_output = BytesIO()
    canvas.print_png(png_output)
    data = png_output.getvalue()
    response = make_response(data)
    response.headers['Content-Type'] = 'image/png'
    return response

@app.route('/hello',methods=["GET","POST"])
def hello():
    image = io.BytesIO()
    x = np.linspace(0, 10)
    y = np.sin(x)
    plt.plot(x, y)
    plt.savefig(image, format='png')
    image.seek(0)
    return send_file(image,
                     attachment_filename="image.png",
                     as_attachment=True)

if __name__ == "__main__":
    app.run(use_reloader=False, threaded=False)