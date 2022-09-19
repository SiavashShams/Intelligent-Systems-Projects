import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

df = load_iris().data


def euclidean_dist(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2, axis=1))


class K_means:
    def __init__(self, K, max_iter=150):  # define our K_means class
        self.max_iter = max_iter
        self.K = K
        self.error = 0
        self.error_iter = []
        self.ratio = 0
        self.mean_cluster = np.zeros(K)
        self.variance_cluster = np.zeros(K)

    def fit(self, X):  # begins clustering
        centroids = X[np.random.choice(X.shape[0], self.K, replace=False), :]  # choose random centroids
        flag = 0
        i = 0
        while i < self.max_iter and flag == 0:  # until algorithm converges or max iteration reached do this
            i = i + 1
            distance = np.zeros((X.shape[0], self.K))
            for k in range(self.K):
                distance[:, k] = euclidean_dist(X, centroids[k, :])
            min_dist = np.nanmin(distance, axis=1)
            ind = []
            for j in range(len(min_dist)):
                ind.append(np.where(distance[j, :] == min_dist[j])[0][0])  # label data according to centroids
            labels = np.array(ind)
            old_centroids = centroids
            centroids = centroids * 0
            for k in range(self.K):
                centroids[k, :] = np.nanmean(X[labels == k, :], axis=0)  # update centroids
            if np.any(old_centroids != centroids):  # check if centroids changed
                flag = 0
            else:
                flag = 1

            dist_from_centeroid = []
            for k in range(self.K):
                dx = norm(X[labels == k] - centroids[k], axis=1)
                dist_from_centeroid.append(dx)
            s = []
            for list in dist_from_centeroid:
                s.append(np.sum(np.square(list)))
            self.error_iter.append(np.sum(s) / len(X[:, 0]))  # calculate cost in each iteration

        dist_from_centeroid = []
        dist_from_other_centeroids = []
        for k in range(self.K):
            dx = norm(X[labels == k] - centroids[k], axis=1)
            dxdx = np.zeros((self.K, len(X[labels == k])))
            for j in range(self.K):
                dxdx[j, :] = norm(X[labels == k] - centroids[j], axis=1)
            dist_from_other_centeroids.append(dxdx)
            dist_from_centeroid.append(dx)
        s = []
        ss = [[] for i in range(self.K)]
        for list in dist_from_centeroid:
            s.append(np.sum(np.square(list)))
        self.error = np.sum(s) / len(X[:, 0])
        z = 0
        for list in dist_from_other_centeroids:
            for list2 in list:
                ss[z].append(np.sum(np.square(list2)))
            z = z + 1
        between_cluster = np.sum(ss) - np.sum(s)
        within_cluster = np.sum(s)
        self.ratio = within_cluster / between_cluster  # Within and Between Cluster Criteria
        for k in range(self.K):  # mean and variance for each cluster
            self.mean_cluster[k] = np.nanmean(X[labels == k])
            self.variance_cluster[k] = np.nanvar(X[labels == k])
            np.nan_to_num(self.mean_cluster, nan=3)
            np.nan_to_num(self.variance_cluster, nan=3)


err = np.zeros((19, 15))
ertd = []
ratio = []
var = []
mean = []
ratio_sameK = [[] for i in range(3)]
ii = 0
for i in [5, 10, 20]:
    for j in range(15):
        km = K_means(K=i,
                     max_iter=150)
        km.fit(df)
        err[i - 2, j] = (km.error)
        err_iter = km.error_iter
        var.append(km.variance_cluster)
        mean.append(km.mean_cluster)
        ratio_sameK[ii].append(km.ratio)
    ii += 1
    plt.plot(err_iter)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()
    ratio.append(km.ratio)
    ertd.append(min(err[i - 2, :]))
var5 = np.array(var[0:10])
mean5 = np.array(mean[0:10])
var10 = np.array(var[15:25])
mean10 = np.array(mean[15:25])
var20 = np.array(var[30:40])
mean20 = np.array(mean[30:40])
mean_ratio_K5 = np.nanmean(ratio_sameK[0])
mean_ratio_K10 = np.nanmean(ratio_sameK[1])
mean_ratio_K20 = np.nanmean(ratio_sameK[2])
var_ratio_K5 = np.nanvar(ratio_sameK[0])
var_ratio_K10 = np.nanvar(ratio_sameK[1])
var_ratio_K20 = np.nanvar(ratio_sameK[2])
var_mean_ratio_df = pd.DataFrame(  # mean and variance of similarity criteria for 15 random starts for each K
    [[mean_ratio_K5, mean_ratio_K10, mean_ratio_K20], [var_ratio_K5, var_ratio_K10, var_ratio_K20]],
    columns=[5, 10, 20], index=["mean", "var"])
print(var_mean_ratio_df)
err = np.zeros((19, 15))
ertd = []
ratio = []
for i in range(2, 21):
    for j in range(15):
        km = K_means(K=i,
                     max_iter=150)
        km.fit(df)
        err[i - 2, j] = (km.error)
        err_iter = km.error_iter
    ratio.append(km.ratio)
    ertd.append(min(err[i - 2, :]))

plt.plot(np.linspace(2, 20, num=19), ertd)
plt.xlabel("K")
plt.ylabel("Cost")
plt.tight_layout();
plt.show()

plt.plot(np.linspace(2, 20, num=19), ratio)
plt.xlabel("K")
plt.ylabel("Within cluster and Between cluster ratio")
plt.tight_layout();
plt.show()
