from tensorflow.keras.optimizers import Adam, RMSprop, SGD


def get_optimizer(**opts):
    """ helper function to get optimizer.

    protocol: Assume opts is a dict with keys matched the correspoding `tf.keras.optimizers.xxx`,
        and with a key `"type"`
    :param opts:
    :return:
    TODO: 可能需要考虑学习率下降的策略
    """
    opts = opts.copy()
    optims_dict = {"adam": Adam, "rmsprop": RMSprop, "sgd": SGD}
    optim_type = optims_dict.get(opts["type"].lower(), optims_dict["sgd"])
    del opts["type"]
    return optim_type(**opts)
