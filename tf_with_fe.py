#contains tensorflow implementation WITH feature engineering

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("seattle-weather.csv")
df['date'] = pd.to_datetime(df['date'])

df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_sin'] = np.sin(2 * np.pi * df['day'] / 30)
df['day_cos'] = np.cos(2 * np.pi * df['day'] / 30)

y = df['weather'].map({"rain": 1, "sun": 2, "fog": 3, "drizzle": 4, "snow": 0}).values.reshape(-1, 1)
df = df.drop(columns=['date', 'weather'])

encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(y)

X = df.values.astype(np.float32)

X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)

n_features = X.shape[1]
n_classes = Y.shape[1]

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(n_features,)),
    tf.keras.layers.Dense(n_classes, activation='softmax', use_bias=False)
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.025),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X, Y, epochs=100, batch_size=32, verbose=0)  # Use verbose=1 for live updates

loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("Test Accuracy:", accuracy)

'''
from sklearn.metrics import confusion_matrix
import numpy as np

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(Y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
print(cm)
'''
