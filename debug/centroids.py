from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

X = np.array([[0, 0], [1, 1], [2, 2]])
centroids = [[0, 0], [2, 2]]
print(X.shape)

v = euclidean_distances(X, centroids)
closest = np.argmin(v, axis=1)
print(v)
print(closest)
