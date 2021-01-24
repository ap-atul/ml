from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


data = pd.read_csv("ecommerce.csv")

X = data.iloc[:, 3:  7]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

print(f"Training samples:: {len(X_train)}")
print(f"Testing samples:: {len(X_test)}")


model = LinearRegression()
model.fit(X_train, y_train)

print(f"Coefficients :: {model.coef_}")
print(f"Intercept :: {model.intercept_}")

pred = model.predict(X_test)

print("MAE::", mean_absolute_error(y_test, pred))
print("MSE::", mean_squared_error(y_test, pred))
print("RMSE::", np.sqrt(mean_squared_error(y_test, pred)))


plt.scatter(pred, y_test)
plt.xlabel("Predictions")
plt.ylabel("Test labels")
plt.tight_layout()
plt.show()

