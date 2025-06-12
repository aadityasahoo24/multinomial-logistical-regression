# model.py

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

nc = 5  # number of classes

# Load and preprocess dataset once
def load_data():
    df = pd.read_csv("seattle-weather.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 30)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 30)

    y = df['weather'].map({"rain": 1, "sun": 2, "fog": 3, "drizzle": 4, "snow": 0})
    df = df.drop(columns=['date', 'weather'])
    X = df.values

    m, n = X.shape
    Y = np.zeros((m, nc))
    Y[np.arange(m), y] = 1

    return X, Y, n


def softmax(z):
    z = np.array(z)
    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def grad_log(theta, X, Y):
    h = softmax(X.dot(theta))
    l = Y * np.log(h + 1e-15)
    g = X.T.dot(Y - h)
    return g, l


def predict(X, theta):
    h = softmax(X @ theta)
    return np.argmax(h, axis=1)


def run_one_model(X_all, Y_all, n, alpha=0.025, epochs=10000):
    # Shuffle
    perm = np.random.permutation(len(X_all))
    X = X_all[perm]
    Y = Y_all[perm]

    # Train-test split
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    theta = np.zeros((n, nc))

    for e in range(epochs):
        g, _ = grad_log(theta, X_train, Y_train)
        theta += alpha / train_size * g

    prediction = predict(X_test, theta)
    results = np.argmax(Y_test, axis=1)
    accuracy = np.mean(prediction == results)

    return accuracy
