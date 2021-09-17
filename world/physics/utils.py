import io

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

plt.ioff()

from utils.text import TextFlag, log


def create_bidir_lstm_layer(batch_size, lstm_units, return_sequences=True, dropout=0.0, stateful=True):
    forward_layer = tf.keras.layers.LSTM(lstm_units, return_sequences=return_sequences, dtype=tf.float32,
                                         dropout=dropout, stateful=stateful)
    backward_layer = tf.keras.layers.LSTM(lstm_units, return_sequences=return_sequences, go_backwards=True,
                                          dropout=dropout, dtype=tf.float32, stateful=stateful)
    return tf.keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer,
                                         input_shape=(batch_size, int(2 * lstm_units)))


def allow_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            log(TextFlag.INFO, f"Physical GPUs {len(gpus)} Logical GPUs{len(logical_gpus)}")
        except RuntimeError as e:
            log(TextFlag.ERROR, e)


def plot3d(data3d, size):
    dx, dy, dz = np.linspace(0, size, size), np.linspace(0, size, size), np.linspace(0, size, size)
    xx, yy = np.meshgrid(dx, dy)
    xx, yy = np.tile(xx[..., np.newaxis], size), np.tile(yy[..., np.newaxis], size)

    xx = xx.flatten()
    yy = yy.flatten()
    zz = np.tile(np.tile(dz[np.newaxis, ...], size), size).flatten()

    # bound colors to the sigmoid
    colors = data3d.numpy().flatten().astype(np.float32)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(xx, yy, zz, s=50, alpha=0.1, edgecolors='w', c=colors, cmap='RdYlBu',
                         vmin=0, vmax=1)

    plt.colorbar(scatter)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Z')

    return fig


def convert_figure_to_img(figure):
    figure.canvas.draw()
    img = np.array(figure.canvas.renderer.buffer_rgba())[:, :, :3]
    return img[np.newaxis, ...]


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def add_to_tensorboard(metrics: dict, step, prefix):
    for key in metrics:
        if metrics[key] is None:
            return

        if "img" in key:
            img = metrics[key][0]
            minimum = tf.reduce_min(img, (0, 1), True)
            maximum = tf.reduce_max(img, (0, 1), True)
            img = (img - minimum) / (maximum - minimum)
            img = tf.expand_dims(img, 0)
            tf.summary.image('image/{}/{}'.format(prefix, key), img, step=step)

        if "scalar" in key:
            for m in metrics[key]:
                tf.summary.scalar('{}/{}'.format(prefix, m.name), m.result().numpy(), step=step)