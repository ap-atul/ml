from matplotlib import pyplot as plt
from kmeans import *

points = [[0.1,0.6], [0.15,0.71], [0.08,0.9], [0.16,0.85], [0.2,0.3], [0.25,0.5], [0.24,0.1], [0.3,0.2]]
centers = [[0.1, 0.6], [0.3, 0.2]]

km = KMeans()
km.fit(points)

# 1. P6 belongs to cluster
print(f"P6 point belongs to cluster:: {km.predict(points[5])}")

# 2. Population of cluster 2
print(f"Population of cluster 2 :: {len(km._classifications[1])}")

# 3. Updated values for the clusters
print(f"Updated clusters :: {list(km._centroids.values())}")

# plotting the cluster and centers
colors = ['r', 'g']
for classification in km._classifications:
    for x, y in km._classifications[classification]:
        plt.scatter(x, y, color=colors[classification], s=20)

for center in km._centroids:
    x, y = km._centroids[center]
    plt.scatter(x, y, color=colors[center], marker="*", s=50)
plt.show()

