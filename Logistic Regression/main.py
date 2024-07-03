import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression

df = pd.read_csv(filepath_or_buffer="Logistic Regression/diabetes2.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

model = LogisticRegression(epochs=1000, learning_rate=0.01)
model.fit(X, y)
y_pred = model.predict(X)

print(round((y == y_pred).mean() * 100, 2), "% accuracy", sep="")

plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="red", label="0")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="blue", label="1")
plt.xlabel("Glucose")
plt.ylabel("Blood Pressure")
plt.legend()
plt.show()
