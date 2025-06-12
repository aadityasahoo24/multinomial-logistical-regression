import numpy as np
from model import load_data, run_one_model

X_all, Y_all, n = load_data()

accuracies = []

for i in range(100):
    acc = run_one_model(X_all, Y_all, n)
    print(acc)
    accuracies.append(acc)

print("All accuracies:", accuracies)
print(f"Mean accuracy over 100 runs: {np.mean(accuracies):.4f}")
print(f"Standard deviation: {np.std(accuracies):.4f}")
