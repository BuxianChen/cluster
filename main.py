# -*- coding: UTF-8 -*-
import argparse
import os

from data.datasets import dataset_clusters

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) == 0:
    print("Not enough GPU hardware devices available")
else:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ == "__main__":
    from models.model import main
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_name", type=str, choices=list(dataset_clusters.keys()), default="mnist",
                        help="dataset name (default: mnist)")
    parser.add_argument("-m", "--model_name", type=str, default="dec",
                        help="model name (default: dec)")
    parser.add_argument("--model_type", type=str, default="allconv",
                        help="network structure, available type (default: allconv):\n"
                             "\tmlp: all fully-connected layers,\n"
                             "\tallconv: all convolutional layers,\n"
                             "\tconv: convolutional and pool layers.")
    parser.add_argument("--norm_type", type=str, default="None",
                        help="type of normalization layer (default: None):\n"
                             "\tNone: no normalization layers,\n"
                             "\tbn: batch normalization layers,\n"
                             "\tnormal: non-trainable batch normalization layers.")
    parser.add_argument("--acti_type", type=str, default="relu",
                        help="type of activation layer")
    parser.add_argument("--pretrain_augment", dest="pretrain_augment", action="store_true",
                        help="use data augmentation in pretrain stage, (default: True)")
    parser.add_argument("--not_pretrain_augment", dest="pretrain_augment", action="store_false",
                        help="not use data augmentation in pretrain stage, (default: False)")
    parser.set_defaults(pretrain_augment=True)
    parser.add_argument("--train_augment", dest="train_augment", action="store_true",
                        help="use data augmentation in train stage, (default: True)")
    parser.add_argument("--not_train_augment", dest="train_augment", action="store_false",
                        help="use data augmentation in train stage, (default: False)")
    parser.set_defaults(train_augment=True)
    # dccbc args
    parser.add_argument("--num_cluster", type=int, default=-1,
                        help="number of clusters, if -1, use ground-truth, (default: -1)")
    parser.add_argument("--embeded_dim", type=int, default=-1,
                        help="encoder's output dims, if -1, embeded_dim==num_cluster, (default: -1)")
    parser.add_argument("--re", type=float, default=0.9,
                        help="reconstruct loss weight")
    parser.add_argument("--cl", type=float, default=0.1,
                        help="dec cluster loss weight")
    parser.add_argument("--cre", type=float, default=0.1,
                        help="fuzzy c-means loss weight")
    parser.add_argument("--dccbc", type=float, default=0.0,
                        help="center-based reconstruct loss weight")

    parser.add_argument("--pretrain_epoch", type=int, default=100)
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--cluster_loss_type", type=str, default="kl")
    parser.add_argument("--reconstruct_loss_type", type=str, default="l2")
    parser.add_argument("--num_repeat_kmeans", type=int, default=20)
    parser.add_argument("--pretrain_lr", type=float, default=0.001)
    parser.add_argument("--pretrain_optimizer_type", type=str, default="adam")
    parser.add_argument("--train_lr", type=float, default=0.001)
    parser.add_argument("--train_optimizer_type", type=str, default="adam")
    parser.add_argument("--not_use_pretrain_model", dest="use_pretrain_model", action="store_false")
    parser.add_argument("--use_pretrain_model", dest="use_pretrain_model", action="store_true")
    parser.set_defaults(use_pretrain_model=True)
    parser.add_argument("--not_save_pretrain_model", dest="save_pretain_model", action="store_false")
    parser.add_argument("--save_pretrain_model", dest="save_pretrain_model", action="store_true")
    parser.set_defaults(save_pretrain_model=False)


    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--buffer_size", type=int, default=70000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.)
    parser.add_argument("--delta", type=float, default=0.001)

    parser.add_argument("--silent", dest="silent", action="store_true")
    parser.add_argument("--not_silent", dest="silent", action="store_false")
    parser.set_defaults(silent=False)
    parser.add_argument("--slevel", type=str, default="debug")
    parser.add_argument("--to_disk", dest="to_disk", action="store_true")
    parser.add_argument("--not_to_disk", dest="to_disk", action="store_false")
    parser.set_defaults(to_disk=True)
    parser.add_argument("--flevel", type=str, default="debug")
    args = parser.parse_args()
    print(args)
    main(args)