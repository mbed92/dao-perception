import tensorflow as tf

from utils.text import TextFlag, log


def run(nn, ds, writer, step, prefix, optimizer=None, eta=None):
    is_training = 'train' in prefix

    # run epoch
    metric_loss = tf.keras.metrics.Mean(name="MeanLoss")
    for obs, act, y_true in ds:
        if is_training:
            with tf.GradientTape() as tape:
                y_pred = nn([obs, act], training=True)
                loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true=y_true, y_pred=y_pred))

            gradients = tape.gradient(loss, nn.trainable_variables)
            optimizer.apply_gradients(zip(gradients, nn.trainable_variables))
        else:
            y_pred = nn([obs, act], training=False)
            loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true=y_true, y_pred=y_pred))

        metric_loss.update_state(loss.numpy())
        step += 1

    # add to the tensorboard
    log(TextFlag.INFO, f'Mean {prefix} loss: {metric_loss.result().numpy()}')
    with writer.as_default():
        tf.summary.scalar(prefix + metric_loss.name, metric_loss.result().numpy(), step=step)
        if is_training:
            tf.summary.scalar(prefix + "lr", eta.value().numpy(), step=step)
    writer.flush()
    return step
