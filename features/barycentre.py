from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
from sklearn.metrics import euclidean_distances


class BaryCentre(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.centroids = []

        for class_ in self.classes_:
            class_indices = np.where(self.y_ == class_)[0]
            class_datas = np.take(self.X_, class_indices, axis=0)
            class_centroid = self._compute_centroid(class_datas)
            self.centroids.append(class_centroid)

        return self

    def predict(self, x):
        check_is_fitted(self)
        x = check_array(x)
        closest = np.argmin(euclidean_distances(x, self.centroids), axis=1)
        return self.classes_[closest]

    def _compute_centroid(self, points):
        return list(np.mean(points, axis=0))