import numpy as np
import tensorflow as tf

from utils.text import TextFlag, log


def reward_haptic(state, action, **kwargs):
    assert "predictive_model" in kwargs.keys()
    assert "y_true" in kwargs.keys()
    assert "steps" in kwargs.keys()
    assert "time_penalty_delta" in kwargs.keys()

    feed = (state, action.to_numpy().reshape((1, -1)))
    feed = [f[np.newaxis, ...] for f in feed]
    y_pred = kwargs["predictive_model"](feed, training=False)
    y_pred = tf.reshape(y_pred, -1)

    reward = 0.0
    if "y_true" in kwargs.keys():
        y_true = tf.convert_to_tensor([v for v in kwargs["y_true"].values()])
        loss = tf.losses.mean_absolute_error(y_true, y_pred)
        reward = 1.0 / (1e-6 + loss)
        reward = reward - kwargs["steps"] * kwargs["time_penalty_delta"]

    else:
        log(TextFlag.WARNING, f"Cannot calculate reward. Equals: {reward}")

    return reward


def reward_from_haptic_net(state, action, **kwargs):
    assert "y_true" in kwargs.keys()
    assert "model" in kwargs.keys()

    feed = (state, np.tile(action[np.newaxis, ...], [1, 2, 1]))
    with tf.GradientTape() as tape:
        y_pred = kwargs["model"](feed, training=True)
        y_true = tf.convert_to_tensor([v for v in kwargs["y_true"].values()])
        loss_no_reg = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true=y_true, y_pred=y_pred))
        l2_reg = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in kwargs["model"].trainable_variables]) * 0.001
        loss_reg = loss_no_reg + l2_reg

    gradients = tape.gradient(loss_reg, kwargs["model"].trainable_variables)
    reward = (1.0 / (tf.abs(loss_no_reg) + 1e-5))
    return reward, gradients, loss_no_reg
