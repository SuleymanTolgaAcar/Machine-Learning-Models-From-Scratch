import matplotlib.pyplot as plt
import pandas as pd

from utils.metrics import R_squared
from utils.math import normalize
from utils.misc import train_test_split

from .multiple_linear_regression import MultipleLinearRegression

df = pd.read_csv("multiple_linear_regression/Advertising and Sales.csv")
df = df.dropna()
df = normalize(df.loc[:, ["TV", "Radio", "Social Media", "Sales"]])
X, y = df.iloc[:, :-1], df.iloc[:, -1]
X_train, y_train, X_test, y_test = train_test_split(X, y)

grad_model = MultipleLinearRegression(
    epochs=1000, learning_rate=0.05, algorithm="gradient_descent"
)
grad_model.fit(X_train, y_train)
grad_y_pred = grad_model.predict(X_test)
grad_r2 = R_squared(y_test, grad_y_pred)
print(f"Gradient Descent R^2: {round(grad_r2, 6)}")
print(f"Gradient Descent b: {grad_model.b.round(6)}")

ols_model = MultipleLinearRegression(algorithm="ordinary_least_squares")
ols_model.fit(X_train, y_train)
ols_y_pred = ols_model.predict(X_test)
ols_r2 = R_squared(y_test, ols_y_pred)
print(f"Ordinary Least Squares R^2: {round(ols_r2, 6)}")
print(f"Ordinary Least Squares b: {ols_model.b.round(6)}")


plt.scatter(y_test, grad_y_pred, color="blue", label="Gradient Descent")
plt.scatter(y_test, ols_y_pred, color="red", label="Ordinary Least Squares")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.legend()
plt.show()
