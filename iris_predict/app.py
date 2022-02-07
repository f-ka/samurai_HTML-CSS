from flask import Flask, render_template, request, flash
from wtforms import Form, FloatField, SubmitField, validators, ValidationError
import numpy as np
from sklearn.externals import joblib

# 学習済みモデルを読み込み利用します
def predict(parameters):
    # ニューラルネットワークのモデルを読み込み
    model = joblib.load('./nn.pkl')
    params = parameters.reshape(1,-1)
    pred = model.predict(params)
    return pred

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

app = Flask(__name__)
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

if __name__ == "__main__":
    app.run()