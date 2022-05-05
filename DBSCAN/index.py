from sklearn.datasets import make_moons, make_circles
from queue import Queue
import numpy as np
import matplotlib.pyplot as plt


class DBSCAN:
    def __init__(self, min_samples=10, r=0.15):
        self.min_samples = min_samples
        self.r = r
        self.X = None
        self.label = None
        self.n_class = 0

    def fit(self, X):
        self.X = X
        self.label = np.zeros(X.shape[0])
        q = Queue()
        for i in range(len(self.X)):
            if self.label[i] == 0:
                q.put(self.X[i])
                if self.X[(np.sqrt(np.sum((self.X - self.X[i]) ** 2, axis=1)) <= self.r) & (self.label == 0)].shape[0] >= self.min_samples:
                    self.n_class += 1
                while not q.empty():
                    p = q.get()
                    neighbors = self.X[(
                        np.sqrt(np.sum((self.X - p) ** 2, axis=1)) <= self.r) & (self.label == 0)]
                    if neighbors.shape[0] >= self.min_samples:
                        mark = (
                            np.sqrt(np.sum((self.X - p) ** 2, axis=1)) <= self.r)
                        self.label[mark] = np.ones(
                            self.label[mark].shape) * self.n_class
                        # print(self.label)
                        for x in neighbors:
                            q.put(x)

    def plot_dbscan_2D(self):
        plt.rcParams['font.sans-serif'] = ["SimHei"]
        plt.rcParams['axes.unicode_minus'] = False
        for i in range(self.n_class+1):
            if i == 0:
                label = '异常数据'
            else:
                label = '第'+str(i) + '类数据'
            plt.scatter(self.X[self.label == i, 0],
                        self.X[self.label == i, 1], label=label)
        plt.legend()
        plt.show()


X, _ = make_circles(n_samples=1000, factor=0.5, noise=0.1)

db = DBSCAN(4, 0.15)
db.fit(X)
db.plot_dbscan_2D()


X, _ = make_moons(n_samples=1000, noise=0.05)
db = DBSCAN(10, 0.15)
db.fit(X)
db.plot_dbscan_2D()
