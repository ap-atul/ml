import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

from lr import *

X = np.array([10, 9, 2, 15, 10, 16, 11, 16])
y = np.array([95, 80, 10, 50, 45, 98, 36, 93])


df = pd.DataFrame({"hours": X, "risk": y})
df.to_csv("example.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50)

print(X_train)

print(f"Training samples :: {len(X_train)}")
print(f"Testing sample :: {len(X_test)}")

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

plt.plot(X_test, pred, marker=".", color='r')
plt.scatter(X_test, y_test, marker="*")
plt.show()

