import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.append("Utility")

from simple_linear_regression import SimpleLinearRegression
from metrics import R_squared

df = pd.read_csv("Simple Linear Regression/Salary_Data.csv", index_col=False)
X = df.iloc[:, 0].values
y = df.iloc[:, 1].values

grad_model = SimpleLinearRegression()
grad_model.fit(X, y, gradient_descent=True, epochs=1000, learning_rate=0.01)
grad_y_pred = grad_model.predict(X)

ols_model = SimpleLinearRegression()
ols_model.fit(X, y, gradient_descent=False)
ols_y_pred = ols_model.predict(X)

print("Gradient Descent Model:")
print("R^2: ", round(R_squared(y, grad_y_pred), 4))
print("Coefficients:", "b0 =", round(grad_model.b0, 4), "b1 =", round(grad_model.b1, 4))

print("\nOrdinary Least Squares Model:")
print("R^2: ", round(R_squared(y, ols_y_pred), 4))
print("Coefficients:", "b0 =", round(ols_model.b0, 4), "b1 =", round(ols_model.b1, 4))

plt.scatter(X, y, color="red")
plt.plot(X, grad_y_pred, color="blue")
plt.title("Experience vs Salary (Gradient Descent Model)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
