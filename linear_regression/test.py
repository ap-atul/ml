import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from lr import *

DATA_FILE = "example.csv"
# DATA_FILE = "dataset.csv"  # body and brain weight

data = pd.read_csv(DATA_FILE, header=None)
X = data.iloc[:, 0]
y = data.iloc[:, 1]

X = np.array(X)
y = np.array(y)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)

#print(f"Training samples :: {len(X_train)}")
#print(f"Testing sample :: {len(X_test)}")

model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)

plt.scatter(X, y, marker="*", color='g')
plt.plot(X, pred, marker=".", color='r')
plt.show()

print(f"RMSE :: {rmse(y, pred)}")
