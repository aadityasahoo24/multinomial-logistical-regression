# Seattle Weather Classification: From-Scratch vs. Frameworks

![Python](https://img.shields.io/badge/Python-3.11.0%2B-blue)
![License](https://img.shields.io/badge/License-GPLv3.0-green)

A comparative study implementing **multi-class logistic regression from scratch** using **NumPy** against **TensorFlow** and **XGBoost** for weather prediction. Built to deepen understanding of core ML principles and feature engineering.

##  Key Insights

| Model                          | Feature Engineering | Mean Accuracy (100 runs) |
|--------------------------------|---------------------|--------------------------|
| **NumPy (From Scratch)**       |  Yes              | 81.56%                   |
| TensorFlow                     |  Yes              | 80.54%                   |
| XGBoost (via TensorFlow)       |  No               | 81.91%                   |

**Surprising Finding**: Our manual implementation with feature engineering came within **0.35%** of XGBoost's performance!

##   Project Structure

- **`data_imp.py`**: Script to import and preprocess the **seattle-weather.csv** dataset from Kaggle.
- **`main.py`**: Contains the **NumPy** and **pandas**-based implementation of the multi-class logistic regression model from scratch with **manual gradient ascent** and **softmax regression**.
- **`res.py`**: Runs the chosen model **100 times**, randomizes the samples, and calculates the **mean accuracy** to evaluate stability and performance.
- **`model.py`**: Used to create the functions used in `res.py`. Allows all 3 models to run 100 times to get the mean accuracy.
- **`seattle-weather.csv`**: A dataset containing historical weather data for Seattle, which includes features like **temperature**, **precipitation**, and **wind**.
- **`tf_with_fe.py`**: TensorFlow implementation of the weather prediction model with **feature engineering** (cyclical encodings for day and month).
- **`tf_wo_fe.py`**: TensorFlow implementation of the weather prediction model **without feature engineering** (using raw weather data).

##  Why Build from Scratch?

While libraries like **scikit-learn**, **TensorFlow**, and **XGBoost** offer pre-built solutions, implementing multi-class logistic regression from scratch provides:

- **Deeper intuition for core ML concepts**: Gain a better understanding of key machine learning principles like gradient ascent, log-likelihood, and numerical stability.
- **Debugging and troubleshooting**: Build your understanding of issues such as softmax instability and numerical overflows, which are difficult to catch in black-box frameworks.
- **Total customization**: Customize the model, loss functions, and gradients, and experiment with different approaches for feature engineering.

##  Performance Comparison

The models were tested on the same dataset and ran 100 randomized training tests. The models' performance is compared based on the **mean accuracy over 100 runs**:

| Model                          | Feature Engineering | Mean Accuracy (100 runs) |
|--------------------------------|---------------------|--------------------------|
| **NumPy (From Scratch)**       |  Yes              | 81.56%                   |
| TensorFlow                     |  Yes              | 80.54%                   |
| XGBoost (via Tensorflow) |  No               | 81.91%                   |

### Key Takeaways:
- **Manual models can be competitive**: The **NumPy-based from-scratch model** performed just **0.35%** below **XGBoost**.
- **Feature engineering matters**: Our **manual NumPy model with feature engineering** outperformed **TensorFlow** with the same features, showcasing the impact of proper feature engineering.

##  Key Challenges & Solutions

### 1. **Softmax Instability**
In logistic regression, the **softmax function** can become unstable due to large values in the logits. To avoid **numerical overflow**, I shifted the logits before exponentiation:

```python
z_shifted = z - np.max(z, axis=1, keepdims=True)  # Critical for stability!
```

### 2. **Gradient Ascent from First Principles**
Instead of using automatic optimizers, I manually derived the gradients for the **log-likelihood function** and computed the gradient ascent for model optimization:

```python
gradient = X.T @ (Y - softmax(X @ theta))  # Vectorized gradient for efficiency
```

### 3. **Feature Engineering Wins**
I applied cyclical encoding for months and days (using sin/cos transformations) to capture periodicity in the data. This encoding helped boost accuracy by +20% over using raw date data.

```python
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_sin'] = np.sin(2 * np.pi * df['day'] / 30)
df['day_cos'] = np.cos(2 * np.pi * df['day'] / 30)
```

##  Lessons Learned

- **Math over Magic**: Frameworks abstract away many implementation details, but building the model from scratch gives you a deeper understanding of the algorithm and helps in debugging.

- **Debugging > Speed**: Catching issues like **softmax overflow** early on saved hours of development time.

- **Feature Engineering = Free Lunch**: Hand-engineered features significantly improved model performance, closing the gap with **XGBoost**.

##  Try It Yourself

The full code is available in this repository. Feel free to experiment with different models, tweak parameters, or contribute optimizations.
