import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

data = pd.read_csv("user_data.csv")
X = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)

tree = DecisionTreeClassifier(criterion='gini', random_state=0)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
print(confusion_matrix(y_test, y_pred))

