from kmeans import *

points = [[0.1,0.6], [0.15,0.71], [0.08,0.9], [0.16,0.85], [0.2,0.3], [0.25,0.5], [0.24,0.1], [0.3,0.2]]
centers = [[0.1, 0.6], [0.3, 0.2]]

km = KMeans()
km.fit(points)
print(km._classifications)
