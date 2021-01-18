import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))

def mean(values):
    return sum(values) / float(len(values))

def variance(values, mean): # sum( (x - mean(x))^2 )
    return sum([(x - mean)** 2 for x in values])

def covariance(x, y, x_mean, y_mean): # sum((x(i) - mean(x)) * (y(i) - mean(y)))
    ret = 0.0
    for i in range(len(x)):
        ret += (x[i] - x_mean) * (y[i] - y_mean)
    return ret

class LinearRegression: # simple linear regression
    def __init__(self):
        self._m = None
        self._b = None

    def fit(self, X, y):
        self._m, self._b = self._coefficients(X, y)
        print(f"Y-intercept :: {self._m}, Bias ::  {self._b}")

    def predict(self, X):
        prediction = list()
        for row in X:
            y = self._b + self._m * row
            prediction.append(y)
        return prediction

    def _coefficients(self, x, y):
        x_mean, y_mean = mean(x), mean(y)
        m = covariance(x, y, x_mean, y_mean) / variance(x, x_mean)
        b = y_mean - m * x_mean
        return m, b
