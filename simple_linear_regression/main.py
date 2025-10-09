import matplotlib.pyplot as plt
import pandas as pd

from .simple_linear_regression import SimpleLinearRegression
from utils.metrics import R_squared

df = pd.read_csv("simple_linear_regression/Salary_Data.csv", index_col=False)
X = df.iloc[:, 0].values
y = df.iloc[:, 1].values

grad_model = SimpleLinearRegression(
    epochs=1000, learning_rate=0.01, algorithm="gradient_descent"
)
grad_model.fit(X, y)
grad_y_pred = grad_model.predict(X)

ols_model = SimpleLinearRegression(algorithm="ordinary_least_squares")
ols_model.fit(X, y)
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
