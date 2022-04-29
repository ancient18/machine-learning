import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np


class KMeans:
    def __init__(self, n_clusters=5, max_iter=2):
        self._n_clusters = n_clusters
        self._X = None
        self._y = None
        self._center = None
        self._max_iter = max_iter

    def fit(self, X):
        self._X = X
        # 随机生成中心点
        print(X.min(axis=0))
        print(X.max(axis=0))
        self._center = np.array([[np.random.uniform(mi, mx) for mi, mx in zip(
            X.min(axis=0), X.max(axis=0))] for _ in range(self._n_clusters)])
        print(self._center.shape)
        step = 0
        # 迭代
        while step < self._max_iter:
            # 求样本点与每个中心点的距离
            distances = np.array(
                [np.sum((X-self._center[i, :])**2, axis=1) for i in range(self._n_clusters)])

            # 样本距离哪个最近中心点
            self._y = np.argmin(distances.T, axis=1)

            print(self._y)

            # 对样本点加权平均计算新的中心点
            self._center = np.array(
                [np.mean(X[self._y == i, :], axis=0) for i in range(self._n_clusters)])
            print(self._center)
            step += 1
            # 显示中间过程
            plt.figure()
            plt.scatter(X[self._y == 0, 0], X[self._y == 0, 1], marker='+')
            plt.scatter(X[self._y == 1, 0], X[self._y == 1, 1], marker='+')
            plt.scatter(X[self._y == 2, 0], X[self._y == 2, 1], marker='+')
            plt.scatter(self._center[0, 0], self._center[0, 1])
            plt.scatter(self._center[1, 0], self._center[1, 1])
            plt.scatter(self._center[2, 0], self._center[2, 1])
            plt.show()

        return self


iris = datasets.load_iris()
X = iris.data[:, 2:]
print(np.sum([1,0,0,0]==0))

# km1 = KMeans(n_clusters=3)
# km1.fit(X)
