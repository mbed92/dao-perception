import os
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
import yaml

import world
from utils.text import TextFlag, log

tf.executing_eagerly()
ENV_CONFIG = yaml.safe_load(open("../config/train_haptic_online.yaml", 'r'))


def start(args):
    myenv = world.environment.pusher.PusherEnvDemo(ENV_CONFIG)
    model = myenv.get_model(ENV_CONFIG)

    # setup an optimization
    os.makedirs(args.logs_path, exist_ok=True)
    eta = tf.Variable(args.lr)
    eta_value = tf.keras.optimizers.schedules.ExponentialDecay(args.lr, args.lr_decay_steps, args.lr_decay_rate)
    eta.assign(eta_value(0))
    optimizer = tf.keras.optimizers.Adam(eta)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_man = tf.train.CheckpointManager(ckpt, args.logs_path, max_to_keep=10)
    train_writer = tf.summary.create_file_writer(args.logs_path + "/train")

    # start training
    metric_loss = tf.keras.metrics.Mean(name="MeanLoss")
    for n_ep in range(args.n_episodes_train):
        for n_act in range(args.n_actions):
            action = world.action.primitives.PushAction.random_sample()
            observations, reward, done, info = myenv.step(action=action)

            # feed network and calculate gradients
            with tf.GradientTape() as tape:
                feed = [f[np.newaxis, ...] for f in info["observations_numpy"]]
                y_pred = model(feed, training=True)
                y_true = np.asarray([v for v in info["haptic"].values()])[np.newaxis, ...]
                loss = tf.keras.losses.mean_squared_error(y_true=y_true, y_pred=y_pred)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            log(TextFlag.INFO, f"Haptic loss: {loss.numpy()}")
            metric_loss.update_state(loss.numpy())

        # do each epoch
        eta.assign(eta_value(0))
        ckpt_man.save()
        myenv.reset()
        model.reset_states()

        # add to the tensorboard
        log(TextFlag.INFO, "Episode: {}, mean loss: {}".format(n_ep, metric_loss.result().numpy()))
        with train_writer.as_default():
            tf.summary.scalar(metric_loss.name, metric_loss.result().numpy(), step=n_ep)
            tf.summary.scalar(eta.name, eta.value().numpy(), step=n_ep)
        train_writer.flush()

    myenv.stop_sim()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--logs-path', type=str, default="./logs_online")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-decay-steps', type=float, default=10)
    parser.add_argument('--lr-decay-rate', type=float, default=0.99)
    parser.add_argument('--n-episodes-train', type=int, default=10000)
    parser.add_argument('--n-actions', type=int, default=30)
    args, _ = parser.parse_known_args()
    world.physics.utils.allow_memory_growth()

    start(args)
