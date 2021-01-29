import pandas as pd
import numpy as np

from knn import *

df = pd.read_csv("train.csv")
X = df.iloc[:, :-1].values
y = df['class'].values

print(f"Training samples :: {len(X)}")
model = Knn()
model.fit(X, y)

df = pd.read_csv("test.csv")
X = df.iloc[:, :-1].values
y = df['class'].values
print(f"Testing sample :: {len(y)}")

pred = model.predict(X)
print(f"Accuracy score :: {accuracy(y, pred)* 100}")
