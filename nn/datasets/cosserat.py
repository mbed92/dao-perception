import os

import numpy as np
import tensorflow as tf

SEQUENCE_LENGTH = 1.0  # seconds
STEP_TIME = 0.2  # seconds
CHOOSE_STEP = int(SEQUENCE_LENGTH / STEP_TIME)


def cosserat_rods_sim_pc(path, content_file, batch_size, load_train=True, load_val=True, load_test=True):
    df = np.load(os.path.join(path, content_file), allow_pickle=True).item()
    datasets = list()

    def crop_seq_x(x):
        return np.asarray([s[:-CHOOSE_STEP:CHOOSE_STEP] for s in x])

    def crop_params(params):
        return np.asarray([[p["density"], p["nu"], p["youngs_modulus"], p["poisson_ratio"]] for p in params])

    def crop_seq_y(y):
        return np.asarray([s[CHOOSE_STEP::CHOOSE_STEP] for s in y])

    if load_train:
        x1 = crop_seq_x(df["train"]["sequences"])
        x2 = crop_params(df["train"]["sequences_params"])
        y = crop_seq_y(df["train"]["sequences"])
        train_ds = tf.data.Dataset.from_tensor_slices((x1, x2, y)).shuffle(1000).batch(batch_size)
        datasets.append(train_ds)
        datasets.append(len(y))

    if load_val:
        x1 = crop_seq_x(df["validation"]["sequences"])
        x2 = crop_params(df["validation"]["sequences_params"])
        y = crop_seq_y(df["validation"]["sequences"])
        val_ds = tf.data.Dataset.from_tensor_slices((x1, x2, y)).shuffle(1000).batch(batch_size)
        datasets.append(val_ds)
        datasets.append(len(y))

    if load_test:
        x1 = crop_seq_x(df["test"]["sequences"])
        x2 = crop_params(df["test"]["sequences_params"])
        y = crop_seq_y(df["test"]["sequences"])
        test_ds = tf.data.Dataset.from_tensor_slices((x1, x2, y)).shuffle(1000).batch(batch_size)
        datasets.append(test_ds)
        datasets.append(len(y))

    return datasets
