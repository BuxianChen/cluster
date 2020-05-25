import h5py
import pickle
import numpy as np
import os

import tensorflow as tf
import tensorflow_datasets as tfds

"""
    所有load_{data_name}函数都返回4个dtype为np.unit8的数组, x是4阶的, y是1阶的
    x shape = (n, H, W, C), y shape = (n, )
    例子: (x_train, y_train), (x_test, y_test) = load_cifar10()
    
    使用方法:
        >>> from data.datasets import available
        >>> from data.datasets import load_merged_data
        >>> # x is a np.array, shape=(n, H, W, C), dtype=np.float32, range=[0, 256)
        >>> # y is a np.array, shape=(n,), dtype=np.int32, range=[0, num_cluster)
        >>> x, y = load_merged_data("mnist")
    
    数据集描述:
        mnist: 手写数字识别库, 60000+10000样本, 10类, shape=(28, 28, 1)
        fashion_mnist: 简单的物体分类库, 60000+10000样本, 10类, shape=(28, 28, 1)
        cifar10: 彩色物体识别库, 50000+10000样本, 10类, shape=(32, 32, 3)
        cifar100: 彩色物体识别库, 50000+10000样本, 100或20类, shape=(32, 32, 3)
        coil100: 彩色物体识别库, 72*100=7200样本, 10类, shape=(128, 128, 3)
        usps: 手写数字识别库, 7291+2007样本, 10类, shape=(16, 16, 1)
        stl10: 彩色物体识别库, 5000+8000样本, 10类, shape=(96, 96, 3)
"""

"""
    YaleB, REUTERS
"""
cifar10_dir = "datasets/cifar10"
cifar100_dir = "datasets/cifar100"
usps_path = "datasets/usps/usps.h5"
stl10_dir = "datasets/stl10"


def load_cifar10(root_dir=cifar10_dir):
    # return 4 uint8 `np.ndarray`s (None, 32, 32, 3), (None, )
    train_files = [os.path.join(root_dir, "data_batch_" + str(i)) for i in range(1, 6)]
    test_file = os.path.join(root_dir, "test_batch")
    # # meta_data, not return
    # meta_file = os.path.join(root_dir, "batches.meta")
    # with open(meta_file, "rb") as fo:
    #     d = pickle.load(fo, encoding="bytes")
    # classes = [str(item, encoding="utf-8") for item in d[b"label_names"]]
    # train data
    x_train, y_train = list(), list()
    for file in train_files:
        with open(file, "rb") as fo:
            d = pickle.load(fo, encoding="bytes")
        x_train.append(np.array(d[b"data"]))
        y_train.append(np.array(d[b"labels"]))
        # b"filenames" maybe used
    x_train = np.concatenate(x_train, axis=0).reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
    y_train = np.concatenate(y_train, axis=0).astype(np.uint8)
    # test data
    with open(test_file, "rb") as fo:
        d = pickle.load(fo, encoding="bytes")
    x_test = np.array(d[b"data"]).reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
    y_test = np.array(d[b"labels"]).astype(np.uint8)
    return (x_train, y_train), (x_test, y_test)


def load_cifar100(root_dir=cifar100_dir, fine=False):
    # return 4 uint8 `np.ndarray`s (None, 32, 32, 3), (None, )
    train_file = os.path.join(root_dir, "train")
    test_file = os.path.join(root_dir, "test")
    # # meta_data, not return
    # meta_file = os.path.join(root_dir, "meta")
    # with open(meta_file, "rb") as fo:
    #     d = pickle.load(fo, encoding="bytes")
    # fine_label_names = [str(item, encoding="utf-8") for item in d[b"fine_label_names"]]
    # coarse_label_names = [str(item, encoding="utf-8") for item in d[b"coarse_label_names"]]
    # train data
    with open(train_file, "rb") as fo:
        d = pickle.load(fo, encoding="bytes")
    x_train = np.array(d[b"data"]).reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
    if fine:
        y_train = np.array(d[b"fine_labels"]).astype(np.uint8)
    else:
        y_train = np.array(d[b"coarse_labels"]).astype(np.uint8)
    # test data
    with open(test_file, "rb") as fo:
        d = pickle.load(fo, encoding="bytes")
    x_test = np.array(d[b"data"]).reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
    if fine:
        y_test = np.array(d[b"fine_labels"]).astype(np.uint8)
    else:
        y_test = np.array(d[b"coarse_labels"]).astype(np.uint8)
    return (x_train, y_train), (x_test, y_test)


def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train.reshape(-1, 28, 28, 1), x_test.reshape(-1, 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)


def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train.reshape(-1, 28, 28, 1), x_test.reshape(-1, 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)


def load_coil100():
    data = tfds.load(name="coil100", split="train")
    x_train, y_train = list(), list()
    for item in data.batch(1024):
        x_train.append(item["image"].numpy())
        y_train.append(item["object_id"].numpy())
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_train = np.char.replace(y_train.astype(str), 'obj', '').astype(np.int8)-1
    x_test = np.zeros((0, *x_train.shape[1:]), dtype=x_train.dtype)
    y_test = np.zeros((0, *y_train.shape[1:]), dtype=y_train.dtype)
    return (x_train, y_train), (x_test, y_test)


def load_usps(path=usps_path):
    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        x_train = (train.get('data')[:] * 255).reshape(-1, 16, 16, 1).astype(np.uint8)
        y_train = (train.get('target')[:]).astype(np.uint8)
        test = hf.get('test')
        x_test = (test.get('data')[:] * 255).reshape(-1, 16, 16, 1).astype(np.uint8)
        y_test = (test.get('target')[:]).astype(np.uint8)
    return (x_train, y_train), (x_test, y_test)


def load_stl10(root_dir=stl10_dir):
    def read_images(filename):
        with open(filename, 'rb') as f:
            images = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(images, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 3, 2, 1))
            return images

    def read_labels(filename):
        with open(filename, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8) - 1
            return labels
    # def read_label_names(filename):
    #     with open(filename, 'r') as f:
    #         label_names = [item.strip() for item in f.readlines()]
    #     return label_names
    x_train = read_images(os.path.join(root_dir, "train_X.bin"))
    x_test = read_images(os.path.join(root_dir, "test_X.bin"))
    y_train = read_labels(os.path.join(root_dir, "train_y.bin"))
    y_test = read_labels(os.path.join(root_dir, "test_y.bin"))
    # label_names = read_label_names(os.path.join(root_dir, "class_names.txt"))
    return (x_train, y_train), (x_test, y_test)


available = {"mnist": load_mnist,
             "fashion_mnist": load_fashion_mnist,
             "cifar10": load_cifar10,
             "cifar100": load_cifar100,
             "coil100": load_coil100,
             "usps": load_usps,
             "stl10": load_stl10}
dataset_clusters = {"mnist": 10,
                    "fashion_mnist": 10,
                    "cifar10": 10,
                    "cifar100": 20,
                    "coil100": 100,
                    "usps": 10,
                    "stl10": 10}


def load_data(name, *args, **kwargs):
    assert name in available, "Data error"
    return available[name](*args, **kwargs)


def load_merged_data(name, *args, **kwargs):
    (x_train, y_train), (x_test, y_test) = load_data(name, *args, **kwargs)
    x = np.concatenate([x_train, x_test], axis=0).astype(np.float32)
    y = np.concatenate([y_train, y_test], axis=0).astype(np.int32)
    return x, y


def load_sample_data(name, num_sample, replace=False, seed=None, *args, **kwargs):
    # num_sample: number of samples per class.
    # replace: True/False
    # if "seed" in kwargs:  # 只是为了装逼
    #     np.random.seed(kwargs["seed"])
    if seed is not None:
        np.random.seed(seed)
    # print(name, num_sample, replace, seed, args, kwargs)
    (x_train, y_train), (x_test, y_test) = load_data(name, *args, **kwargs)
    x = np.concatenate([x_train, x_test], axis=0).astype(np.float32)
    y = np.concatenate([y_train, y_test], axis=0).astype(np.int32)
    labels = np.unique(y)
    x_sample = list()
    y_sample = list()
    for label in labels:  # just for code robust
        idx = np.where(y == label)[0]
        idx_sample = np.random.choice(idx, size=num_sample, replace=replace)
        x_sample.append(x[idx_sample])
        y_sample.append(np.ones(num_sample, dtype=np.int32) * label)
    x_sample = np.concatenate(x_sample, axis=0)
    y_sample = np.concatenate(y_sample, axis=0)
    return x_sample, y_sample
