from decision import *
import numpy as np

X_train = np.array([[1.0, 3.2, 4.4, 5.2],[1.0, 1.9, 2.5, 5.2]])
y_train = [1, 0]

X_test = [[1.0, 1.8, 2.4, 5.1]]
y_test = [[0]]

classifier = DecisionTree()
classifier.fit(np.array(X_train), np.array(y_train))
