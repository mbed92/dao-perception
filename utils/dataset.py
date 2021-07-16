import numpy as np
import tensorflow as tf


def load(file_path, batch_size=1, split_ratio=-1):
    with open(file_path, 'rb') as f:
        dataset = np.load(f, allow_pickle=True).item()  # retrieve dict object
    assert len(dataset['observations']) == len(dataset['actions']) == len(dataset['y'])

    obs = np.asarray(dataset['observations'])
    act = np.asarray(dataset['actions'])
    y = np.asarray(dataset['y'])
    dataset_size = len(dataset['y'])

    # generate dataset
    ds = list()
    if 0.0 < split_ratio < 1.0 and dataset_size > 1:

        # split and load
        split_idx = int(split_ratio * dataset_size)
        obs_train, obs_val = obs[:split_idx], obs[split_idx:]
        act_train, act_val = act[:split_idx], act[split_idx:]
        y_train, y_val = y[:split_idx, 0], y[split_idx:, 0]

        train_tf_ds = tf.data.Dataset.from_generator(
            lambda: iter(zip(obs_train, act_train, y_train)),
            output_types=(tf.float32, tf.float32, tf.float32)) \
            .batch(batch_size) \
            .shuffle(dataset_size)

        val_tf_ds = tf.data.Dataset.from_generator(
            lambda: iter(zip(obs_val, act_val, y_val)),
            output_types=(tf.float32, tf.float32, tf.float32)) \
            .batch(batch_size) \
            .shuffle(dataset_size)

        ds.append(train_tf_ds)
        ds.append(val_tf_ds)

    else:

        # load all
        tf_ds = tf.data.Dataset.from_generator(
            lambda: iter(zip(obs, act, y[:, 0])),
            output_types=(tf.float32, tf.float32, tf.float32)) \
            .batch(batch_size) \
            .shuffle(dataset_size)

        ds = tf_ds

    return ds
