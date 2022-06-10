from sklearn.datasets import make_moons, make_circles
import numpy as np
import matplotlib.pyplot as plt


def DBSCAN(x, eps, MinPts):
    n = x.shape[0]
    # 多创建一列作为visited标志
    column = np.array([-1]*n)
    x = np.c_[x, column]
    # 簇类别
    cluster_id = -1
    # 簇种类的数组
    cluster_id_list = []
    for i in range(n):
        # 如果该点已经被访问过，则跳过
        if x[i, -1] != -1:
            continue
        cluster_id += 1
        x[i, -1] = cluster_id
        cluster_id_list.append(cluster_id)
        # 寻找邻域
        neighbor_idx = region_query(x, x[i], eps)
        if len(neighbor_idx) < MinPts:
            x[i, -1] = -1
            cluster_id_list.remove(cluster_id)
            cluster_id -= 1
        else:
            # 扩展簇
            expand_cluster(x, neighbor_idx, i, cluster_id, eps, MinPts)
    return x, cluster_id_list


# 寻找邻域
def region_query(x, i, eps):
    neighbor_idx = []
    for j in range(x.shape[0]):
        # 如果距离小于阈值，则加入邻域
        if np.sqrt(np.sum((i[:-1] - x[j, :-1])**2)) <= eps:
            neighbor_idx.append(x[j])
    return neighbor_idx


# 扩展簇
def expand_cluster(x, neighbor_idx, i, cluster_id, eps, MinPts):
    for j in neighbor_idx:
        if j[-1] == -1:
            j[-1] = cluster_id
            neighbor_idx_ = region_query(x, j, eps)
            if len(neighbor_idx_) >= MinPts:
                # 如果邻域内的点数大于阈值，则将邻域加入邻域
                neighbor_idx.extend(neighbor_idx_)


# X, _ = make_circles(n_samples=1000, factor=0.1, noise=0.1)


X, _ = make_moons(n_samples=1000, noise=0.05)

x, cluster_id_list = DBSCAN(X, 0.2, 10)

print(cluster_id_list)

# # 将结果可视化
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
plt.scatter(x[:, 0], x[:, 1],c=x[:,-1])
plt.show()