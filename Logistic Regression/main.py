import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression

df = pd.read_csv(filepath_or_buffer="Logistic Regression/diabetes2.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

model = LogisticRegression()
model.fit(X, y, epochs=1000, learning_rate=0.01)
y_pred = model.predict(X)

print(round((y == y_pred).mean() * 100, 2), "% accuracy", sep="")
