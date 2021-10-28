import tensorflow as tf


def absoulte(y_true, y_hat):
    return tf.keras.losses.mean_absolute_error(y_true, y_hat)


def standardize(x, axis=1):
    m, v = tf.nn.moments(x, axes=axis, keepdims=True)
    return (x - m) / (tf.sqrt(v + 1e-6) + 1e-6), m, v


def prepare_input_rod_data(x, x_params, y, augment=False, axis=-1):
    x_target = y[:, -1][:, tf.newaxis]
    x, m, v = standardize(x, axis)

    if augment:
        x += tf.random.normal(tf.shape(v), mean=0.0, stddev=0.1 * tf.sqrt(v + 1e-5))

    return x, x_target, x_params, y
