# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from skfuzzy import cmeans


def kmeans(features, num_cluster, num_repeat=20):
    """
    Args:
        features (np.array): shape=(n, *other_dimension)
        num_cluster (int): 聚类簇数
        num_repeat (int): 重复次数

    Returns:
        y_pred (np.array): shape=(n,), dtype=np.int32, 预测类别
        centroid (np.array): shape=(num_cluster, *other_dimension), 中心点特征

    """
    input_shape = features.shape
    model = KMeans(n_clusters=num_cluster, n_init=num_repeat)
    model.fit(features.reshape((input_shape[0], -1)))
    y_pred = model.predict(features.reshape((input_shape[0], -1))).astype(np.int32)
    centroid = model.cluster_centers_.reshape((num_cluster, *input_shape[1:]))
    return y_pred, centroid


def fuzzy_kmeans(features, num_cluster, m=1.1, num_repeat=20):
    """
    Args:
        features (np.array): shape=(n, *other_dimension)
        num_cluster (int): 聚类簇数
        m (float): 严格大于1的指数
        num_repeat (int): 重复次数

    Returns:
        best_y_pred (np.array): shape=(n,), dtype=np.int32, 预测类别
        best_centroid (np.array): shape=(num_cluster, *other_dimension), 中心点特征

    """
    input_shape = features.shape
    best_y_pred = None
    best_centroid = None
    best_loss = None
    for i in range(num_repeat):
        returns = cmeans(data=features.reshape((input_shape[0], -1)).T, c=num_cluster,
                         m=m, error=1e-4, maxiter=300)
        # 中心, 划分矩阵, 初始矩阵, 距离矩阵, 历史损失, 迭代次数, 好坏程度
        cntr, u, u0, d, jm, p, fpc = returns
        y_pred = np.argmax(u, axis=0)
        if best_y_pred is None or jm[-1] < best_loss:
            best_y_pred = y_pred
            best_centroid = cntr.T
            best_loss = jm[-1]
    best_centroid = best_centroid.reshape((num_cluster, *input_shape[1:]))
    return best_y_pred, best_centroid


def tsne(features):
    return TSNE().fit_transform(features)
