from flask import Flask, render_template, request, flash
from soupsieve import select
from wtforms import Form, FloatField, SubmitField, validators, ValidationError, SelectField
import numpy as np
from sklearn.externals import joblib

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

if __name__ == "__main__":
    app.run()