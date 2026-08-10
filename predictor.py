from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

df = pd.read_csv('train.csv', index_col='id')

X = df.drop(columns=['PitNextLap', 'PitStop', 'RaceProgress'], axis=1)
y = df['PitNextLap']

categories = ["Driver", "Compound", "Race"]
X = pd.get_dummies(X, columns=categories)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)

prediction = model.predict(X_test)
print(prediction)

#print(accuracy_score(y_test, prediction))
