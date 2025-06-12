#without any explicit feature engineering

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("seattle-weather.csv")

weather_map = {"snow": 0, "rain": 1, "sun": 2, "fog": 3, "drizzle": 4}
df['weather'] = df['weather'].map(weather_map)

df = df.dropna()

X = df[['precipitation', 'temp_max', 'temp_min', 'wind']].values
y = df['weather'].values  # already mapped to 0 to 4

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    objective='multi:softmax',
    num_class=5,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

#print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=weather_map.keys()))
