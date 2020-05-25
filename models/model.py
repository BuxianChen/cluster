# -*- coding: UTF-8 -*-
# 此份代码为论文检测稿代码, 对DEC等模型进行的修改
from datetime import datetime
import os
import time

import numpy as np
import tensorflow as tf

from data.datasets import load_merged_data, dataset_clusters, load_sample_data
from data.data_transforms import Transformer
from models.autoencoders import get_backbone
from models.baselines import kmeans
from utils.logger import Logger
from utils.losses import losses
from utils.metrics import metrics
from utils.optimizers import get_optimizer


class ClusterLayer(tf.keras.layers.Layer):
    def __init__(self, num_cluster: int, alpha=1.):
        super(ClusterLayer, self).__init__()
        self.num_cluster = num_cluster
        self.alpha = alpha  # float(alpha)
        self.encoded_dim = None
        self.centroid = None

    def build(self, input_shape):
        self.encoded_dim = input_shape[1:]
        self.centroid = self.add_weight(shape=(self.num_cluster, *self.encoded_dim))
        self.built = True

    def load_weights(self, centroid):
        if not self.built:
            self.build(tf.shape(centroid))
        self.centroid.assign(centroid)


    def call(self, inputs, **kwargs):
        """

        Args:
            inputs (tensor): (b, *self.encoded_dim)
            **kwargs:
                training: True/False, no effect.
        Returns:
            soft_assignment: (b, self.num_cluster)
        """
        inputs = tf.reshape(inputs, (tf.shape(inputs)[0], -1))
        flatten_centroid = tf.reshape(self.centroid, (tf.shape(self.centroid)[0], -1))
        inner = tf.linalg.matmul(inputs, tf.transpose(flatten_centroid))
        l2_dist_square = tf.math.reduce_sum(tf.math.pow(inputs, 2), axis=1, keepdims=True) + tf.math.reduce_sum(
            tf.math.pow(flatten_centroid, 2), axis=1) - 2. * inner
        kernel = tf.math.pow(1. + l2_dist_square / self.alpha, - (self.alpha + 1) / 2)
        soft_assignment = kernel / tf.math.reduce_sum(kernel, axis=1, keepdims=True)
        return soft_assignment


class Pretrainer(object):
    def __init__(self, model, optimizer, loss_fn=losses["l2"]):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @tf.function
    def step(self, batch_x):
        with tf.GradientTape() as tape:
            encoded = self.model["encoder"](batch_x, training=True)
            decoded = self.model["decoder"](encoded, training=True)
            loss = tf.math.reduce_mean(self.loss_fn(batch_x, decoded))
        trainable_variables = [*model["encoder"].trainable_variables,
                               *model["decoder"].trainable_variables]
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))
        return loss


class Trainer(object):
    def __init__(self, model, optimizer, re, cl, cre, dccbc):
        # re: 重构误差权重, cl: 聚类损失权重, cre: 基于中心cmeans/基于中心重构
        self.model = model
        self.optimizer = optimizer
        self.re = re
        self.cl = cl
        self.cre = cre
        self.dccbc = dccbc
        # self.sim = sim

    @tf.function
    def step(self, batch_x, p, q):
        # p为目标分布, q为当前预测分布

        # sim_true = q/np.sqrt(np.sum(q ** 2, axis=1, keepdims=True))
        # sim_true = np.dot(sim_true, sim_true.T)
        # sim_true = (sim_true>0.8).astype(np.float32)

        index = tf.math.argmax(q, axis=1, output_type=tf.dtypes.int32)
        obj_encoded = tf.gather_nd(params=cluster_layer.centroid, indices=tf.reshape(index, (-1, 1)))

        with tf.GradientTape() as tape:
            encoded = model["encoder"](batch_x, training=True)
            soft_assignment = model["cluster_layer"](encoded, training=True)  # q
            decoded = model["decoder"](encoded, training=True)
            # sim_pred = DotDist()(Adaptive()(soft_assignment))

            cl_loss = cluster_loss_fn(p, soft_assignment)
            re_loss = reconstruct_loss_fn(batch_x, decoded)
            # (K, dim)
            dist_mat = tf.reduce_mean((tf.expand_dims(tf.reshape(encoded, (-1, 10)), axis=1) -
                                       tf.reshape(model["cluster_layer"].centroid, (-1, 10))) ** 2,
                                      axis=2)
            cre_loss = tf.reduce_sum(dist_mat * (q ** 2), axis=1)
            # sim_loss = K.binary_crossentropy(sim_true, sim_pred)
            dccbc_loss = reconstruct_loss_fn(model["decoder"](obj_encoded), decoded)

            cl_loss = tf.math.reduce_mean(cl_loss)
            re_loss = tf.math.reduce_mean(re_loss)
            cre_loss = tf.math.reduce_mean(cre_loss)
            # sim_loss = tf.math.reduce_mean(sim_loss)
            dccbc_loss = tf.math.reduce_mean(dccbc_loss)

            loss = self.cre * cre_loss + self.cl * cl_loss + self.re * re_loss + self.dccbc * dccbc_loss

        trainable_variables = [*model["encoder"].trainable_variables,
                               *model["decoder"].trainable_variables,
                               *model["cluster_layer"].trainable_variables]
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))
        return loss, cl_loss, cre_loss, re_loss, dccbc_loss, soft_assignment


def initialize_model(model, dataset, num_repeat_kmeans=20):
    """
    Args:
        model (dict): contains `encoder, decoder, cluster_layer`
        dataset (tf.data.Dataset): shuffled and batched dataset
        num_repeat_kmeans: times of random initialization

    Returns:
        y_true (np.array): shape = (n,), true labels.
        y_pred (np.array): shape = (n,), predict labels.
        soft_assignment (np.array): shape = (n, num_cluster), predict probs
    """
    encoder, decoder, cluster_layer = model["encoder"], model["decoder"], model["cluster_layer"]
    inputs = []
    y_true = []
    for item in dataset:
        encoded = encoder(item["feature"])
        y_true.append(item["label"].numpy())
        inputs.append(encoded.numpy())
    inputs = np.concatenate(inputs, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    # y_pred, centroid = fuzzy_kmeans(features=inputs, num_cluster=cluster_layer.num_cluster,
    #                                 num_repeat=num_repeat_kmeans)
    y_pred, centroid = kmeans(features=inputs, num_cluster=cluster_layer.num_cluster,
                              num_repeat=num_repeat_kmeans)
    cluster_layer.load_weights(centroid)
    soft_assignment = cluster_layer(inputs, training=False).numpy()
    return y_true, y_pred, soft_assignment


@tf.function
def test_step(model, item):
    # returns: indices (tf.tensor), shape=(batch_size,), predict cluster labels
    inputs = item["feature"]
    encoded = model["encoder"](inputs, training=False)
    soft_assignment = model["cluster_layer"](encoded, training=False)
    indices = tf.math.argmax(soft_assignment, axis=1, output_type=tf.dtypes.int32)
    return indices, soft_assignment


def test(model, test_dataset, return_probs=False):
    y_true, y_pred = list(), list()
    soft = list()
    for item in test_dataset:
        label = item["label"]
        pred, soft_assignment = test_step(model, item)
        y_true.append(label.numpy())
        y_pred.append(pred.numpy())
        soft.append(soft_assignment.numpy())
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    soft = np.concatenate(soft, axis=0)
    if return_probs:
        return y_true, y_pred, soft
    else:
        return y_true, y_pred


def stop_criterion(y_pred_last, y_pred, delta):
    """
    Args:
        y_pred_last (np.array or None): last prediction, int dtype, shape=(n,)
        y_pred (np.array): current prediction, int dtype, shape=(n,)
        delta (float): if the difference fraction between `y_pred_last` and `y_pred`
            is more than delta, stop training

    Returns:
        flag:
            True: should stop training
            False: should continue training
        rate (float): changed rate
        num (float): changed num
    """
    flag = False
    rate = None
    num = None
    if y_pred_last is not None:
        num = np.sum(y_pred_last != y_pred)
        rate = num / y_pred.shape[0]
        if rate < delta:
            flag = True
    return flag, rate, num


def get_params(args):
    np.set_printoptions(precision=3, suppress=True)
    global model_name, data_name, model_type, norm_type, acti_type
    global pretrain_augment, train_augment
    global num_cluster, embeded_dim, silent, slevel, to_disk, flevel, time_stamp, log_file
    global save_model_dir, print_every, buffer_size, batch_size, alpha
    global pretrain_lr, pretrain_optimizer_type
    global pretrain_epoch, train_lr, train_optimizer_type, maxiter, interval
    global cluster_loss_type, reconstruct_loss_type
    global num_repeat_kmeans, use_pretrain_model, save_pretrain_model, delta
    global re, cre, cl, dccbc
    # ==================== some other hyper-parameters ===================
    model_name = args.model_name  # "dec"
    data_name = args.data_name  # "mnist"
    model_type = args.model_type  # "conv" or "all_conv" or "mlp"
    norm_type = args.norm_type if args.norm_type is not "None" else None  # None or "bn" or "normal"
    acti_type = args.acti_type  # "relu"
    pretrain_augment = args.pretrain_augment  # True
    train_augment = args.train_augment # True
    model_name = "_".join([model_name, model_type])  # "dec_mnist"
    num_cluster = dataset_clusters[data_name] if args.num_cluster < 0 else args.num_cluster  # 10
    embeded_dim = num_cluster if args.embeded_dim < 0 else args.embeded_dim  # 10
    silent = args.silent  # False
    slevel = args.slevel  # "debug" or "info"
    to_disk = args.to_disk  # True
    flevel = args.flevel  # "debug" or "info"
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_file = f"results/log/{model_name}_{data_name}_{time_stamp}.txt"
    save_model_dir = "results/model/"
    print_every = args.print_every  # 10
    buffer_size = args.buffer_size  # 70000
    batch_size = args.batch_size  # 256

    alpha = args.alpha  # 1
    pretrain_lr = args.pretrain_lr  # 0.001
    pretrain_optimizer_type = args.pretrain_optimizer_type  # "rmsprop"
    pretrain_epoch = args.pretrain_epoch  # 100
    train_lr = args.train_lr  # 0.001
    train_optimizer_type = args.train_optimizer_type  # "rmsprop"
    maxiter = args.maxiter
    interval = args.interval
    cluster_loss_type = args.cluster_loss_type  # "kl"
    reconstruct_loss_type = args.reconstruct_loss_type  # "l2"
    num_repeat_kmeans = args.num_repeat_kmeans  # 20
    use_pretrain_model = args.use_pretrain_model  # False
    save_pretrain_model = args.save_pretrain_model  # True

    delta = args.delta  # 0.001

    re, cre, cl, dccbc = args.re, args.cre, args.cl, args.dccbc

    # ==================== some settings ====================
    global logger, feature, label, train_dataset, test_dataset, transformer
    global sample_x, sample_y, pretrain_optimizer, train_optimizer
    global encoder, decoder, cluster_layer, model, cluster_loss_fn, reconstruct_loss_fn
    global encoder_path, decoder_path

    logger = Logger(silent=silent, slevel=slevel, to_disk=to_disk,
                    log_file=log_file, flevel=flevel)
    feature, label = load_merged_data(data_name)
    feature = feature / 255.0


    dataset = tf.data.Dataset.from_tensor_slices({"feature": feature,
                                                  "label": label,
                                                  "idx": np.arange(label.shape[0])})
    train_dataset = dataset.shuffle(buffer_size).batch(batch_size)
    test_dataset = dataset.batch(batch_size)
    transformer = Transformer(data_name, batch_size)

    sample_x, sample_y = load_sample_data(data_name, 100)
    sample_x = sample_x / 255.

    pretrain_optimizer = get_optimizer(type=pretrain_optimizer_type,
                                       learning_rate=pretrain_lr)
    train_optimizer = get_optimizer(type=train_optimizer_type,
                                    learning_rate=train_lr)
    pretrain_augment_string = "augment" if pretrain_augment else "no_augment"
    encoder_path = "_".join([data_name, model_type, pretrain_augment_string, "encoder.h5"])
    encoder_path = os.path.join(save_model_dir, encoder_path)
    decoder_path = "_".join([data_name, model_type, pretrain_augment_string, "decoder.h5"])
    decoder_path = os.path.join(save_model_dir, decoder_path)
    if use_pretrain_model:
        logger.info(f"Use pretrained model: {encoder_path}, {decoder_path}")
        encoder = tf.keras.models.load_model(encoder_path)
        decoder = tf.keras.models.load_model(decoder_path)
        pretrain_epoch = 0
    else:
        logger.info("Warning: train Autoencoder from scratch")
        encoder, decoder = get_backbone(data_name, embeded_dim=embeded_dim,
                                        model_type=model_type, norm_type=norm_type,
                                        acti_type=acti_type)

    cluster_layer = ClusterLayer(num_cluster, alpha)
    model = {"encoder": encoder, "decoder": decoder, "cluster_layer": cluster_layer}

    cluster_loss_fn = losses[cluster_loss_type]
    reconstruct_loss_fn = losses[reconstruct_loss_type]


def main(args):
    re_loss_list = []
    get_params(args)
    global feature
    # ==================== log ====================
    logger.info("=" * 20 + " Setting " + "=" * 20)
    logger.info(args)
    # ==================== do some check here ====================
    model["encoder"].summary(print_fn=logger.info)
    model["decoder"].summary(print_fn=logger.info)
    # ==================== train ====================
    logger.info("=" * 20 + " Begin Training " + "=" * 20)
    # ==================== train AE ====================
    time1 = time.time()
    logger.info("=" * 20 + " Begin Pretrain AutoEncoder " + "=" * 20)
    pretrainer = Pretrainer(model, pretrain_optimizer, reconstruct_loss_fn)
    for epoch in range(pretrain_epoch):
        if (epoch + 1) % 5 == 0:
            logger.info("do kmeans")
            inputs = model["encoder"].predict(feature)
            y_pred, centroid = kmeans(features=inputs, num_cluster=num_cluster,
                                      num_repeat=num_repeat_kmeans)
            results = {key: metrics[key](label, y_pred) for key in metrics.keys()}
            logger.info(" ".join([key + f": {results[key]: .3f}," for key in sorted(results.keys())]))
        t1 = time.time()
        for i, item in enumerate(train_dataset):
            x_batch = item["feature"].numpy()
            x_batch = transformer.transform(x_batch) if pretrain_augment else x_batch
            loss = pretrainer.step(x_batch)
            if (i + 1) % print_every == 0:
                logger.info(f"[epoch: {epoch + 1:3d}]\t[iteration: {i + 1:3d}]\t[loss: {loss:.3f}]")
        t2 = time.time()
        logger.info(f"[epoch {epoch + 1:3d} finished]\t[train time: {t2 - t1: .3f}s]")
    if save_pretrain_model:
        if not os.path.exists(save_model_dir) and save_model_dir:
            os.makedirs(save_model_dir)
        logger.info(f"Save pretrained model: {encoder_path}, {decoder_path}")
        model["encoder"].save(encoder_path)
        model["decoder"].save(decoder_path)
    # ==================== k-means ===================
    time2 = time.time()
    logger.info("=" * 20 + " Use k-means algorithm to initialize " + "=" * 20)
    y_true, y_pred, init_assignment = initialize_model(model, test_dataset, num_repeat_kmeans)
    logger.info("more infomation")
    for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
        logger.info(f">{th}: {100*sum(np.max(init_assignment, axis=1) > th) / y_true.shape[0]:.2f}%")
    results = {key: metrics[key](y_true, y_pred) for key in metrics.keys()}
    logger.info(" ".join([key + f": {results[key]: .3f}," for key in sorted(results.keys())]))
    # ==================== train DEC ====================
    time3 = time.time()
    logger.info("=" * 20 + " Begin Training " + "=" * 20)
    trainer = Trainer(model, train_optimizer, re, cl, cre, dccbc)
    y_pred_last = y_pred

    iter = 0
    idx = 0
    idxmax = feature.shape[0] // batch_size
    q = model["cluster_layer"](model["encoder"].predict(feature)).numpy()
    p = (q ** 2) / (np.sum(q, axis=0) ** 1)
    p = p / np.sum(p, axis=1, keepdims=True)
    while iter < maxiter:
        # plot_tsne(sample_x, sample_y, model["encoder"], iter)
        # if iter % (interval * 10) == 0:
        #     pass
        #     lambda_origin *= 0.95
        x_batch = feature[idx * batch_size:(idx + 1) * batch_size]
        x_batch = transformer.transform(x_batch) if train_augment else x_batch
        p_batch = p[idx*batch_size:(idx+1)*batch_size]
        q_batch = q[idx*batch_size:(idx+1)*batch_size]
        loss, cl_loss, cre_loss, re_loss, dccbc_loss, soft_assignment = \
            trainer.step(x_batch, p_batch, q_batch)
        if (iter + 1) % print_every == 0:  #print_every
            logger.info(f"[iteration: {iter + 1:3d}], "
                        f"[loss: {loss: .3f}]\n[cluster_loss: {cl_loss: .3f}], "
                        f"[cmeans_loss: {cre_loss: .3f}], "
                        f"[reconstruct_loss: {re_loss: .3f}], "
                        f"[dccbc_loss: {dccbc_loss: .3f}]")
            re_loss_list.append(float(re_loss))
        idx = (idx + 1) % idxmax
        if (iter + 1) % interval == 0:
            #####
            tmp = np.arange(feature.shape[0])
            np.random.shuffle(tmp)
            feature = feature[tmp]
            #####
            q = model["cluster_layer"](model["encoder"].predict(feature)).numpy()
            p = (q ** 2) / (np.sum(q, axis=0)**1)
            p = p / np.sum(p, axis=1, keepdims=True)

            y_true, y_pred, y_prob = test(model, test_dataset, True)
            for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
                logger.info(f">{th}: {100 * sum(np.max(y_prob, axis=1) > th) / y_true.shape[0]:.2f}%")
            results = {key: metrics[key](y_true, y_pred) for key in metrics.keys()}
            logger.info(" ".join([key + f": {results[key]: .3f}," for key in sorted(results.keys())]))
            flag, changed_rate, changed_num = stop_criterion(y_pred_last, y_pred, delta)
            logger.info(f"[count predict class: {np.bincount(y_pred) / y_pred.shape[0]}]"
                        f"[changed labels: {changed_num}]\t"
                        f"[changed_rate: {changed_rate: .4f}]")
            if flag and iter > 0:
                logger.info(f"stoped at iter {iter}, change is small than {delta}.")
                break
            y_pred_last = y_pred
        iter += 1
    # ==================== summary ====================
    time4 = time.time()
    logger.info(f"Complete.\n"
                f"pretrain time: {time2 - time1: .2f}\n"
                f"k-means time: {time3 - time2: .2f}\n"
                f"train time: {time4 - time3: .2f}")

    # fig.savefig("results/figure/tsne.png")
    # plt.close(fig)
    return model, re_loss_list