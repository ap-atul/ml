import numpy as np

def _euclidean(points, center):
    return np.sqrt(sum((pnt - cntr) ** 2 for pnt, cntr in zip(points, center)))

class KMeans:
    def __init__(self, k=2, tol=0.001, epochs=30):
        self._k, self._tol, self._epochs = k, tol, epochs
        self._classifications, self._centroids = dict(), dict()

    def fit(self, data, init_centroids=None):
        if init_centroids is None:
            self._centroids = dict(enumerate(data[0: self._k]))
        else:
            self._centroids = dict(enumerate(init_centroids[0: self._k]))

        for _ in range(self._epochs):
            for i in range(self._k):
                self._classifications[i] = list()

            for row in data:
                distances = [_euclidean(row, self._centroids[centroid]) for centroid in self._centroids]
                self._classifications[distances.index(min(distances))].append(row)

            prev_centroids = dict(self._centroids)
            for classification in self._classifications:
                self._centroids[classification] = np.average(self._classifications[classification], axis=0)

    def predict(self, data):
        distances = [np.linalg.norm(data - self._centroids[centroid]) for centroid in self._centroids]
        return distances.index(min(distances))
