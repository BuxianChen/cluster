"""metrics functional API

    典型用法:
        >>> from utils.metrics import metrics
        >>> y_true, y_pred = ...
        >>> metrics["nmi"](y_true, y_pred)

"""
from functools import partial

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

nmi = partial(normalized_mutual_info_score, average_method='arithmetic')
ari = adjusted_rand_score


def acc(y_true, y_pred, return_idx=False):
    """y_true, y_pred, all (N,) np.int64 numpy.array"""
    assert y_pred.shape == y_true.shape
    d = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((d, d), dtype=np.int64)
    # d_true, d_pred = np.max(y_true) + 1, np.max(y_pred) + 1
    # w = np.zeros((d_true, d_pred), dtype=np.int64)
    for i in range(y_true.size):
        w[y_true[i], y_pred[i]] += 1
    idx_true, idx_pred = linear_sum_assignment(np.max(w) - w)
    # idx_true = np.arange(d), idx_pred = shuffled `np.arange(d)`
    # print(idx_true, idx_pred)
    accuracy = w[idx_true, idx_pred].sum() / y_true.size
    if return_idx:
        return accuracy, idx_pred
    else:
        return accuracy

def label_align(y_pred, idx_pred):
    # y_pred.shape=(n,), idx_pred.shape = (k,), where `idx_pred` is a permutation of `np.arange(k)`
    # the map is: for any i in [0, k), [label_true: i] -> [label_pred: idx_pred[i]]
    # returns: aligned predict labels, shape=(n, )
    aligned_pred = np.zeros_like(y_pred)
    for i in range(idx_pred.size):
        aligned_pred[y_pred==idx_pred[i]] = i
    return aligned_pred

metrics = {"nmi": nmi, "ari": ari, "acc": acc}
