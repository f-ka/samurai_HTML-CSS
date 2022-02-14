import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

train_data = pd.read_csv("./project/titanic/train.csv")
test_data = pd.read_csv("./project/titanic/test.csv")

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
model.predict_proba(X_test)[:,1]

joblib.dump(model, "./project/titanic/Titanic_nn.pkl", compress=True)
print("Your submission was successfully saved!")
