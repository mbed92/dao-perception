import os
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
import yaml

import world

tf.executing_eagerly()
NUM_EPISODES = 10
NUM_ACTIONS = 10
ENV_CONFIG = yaml.safe_load(open("../config/env.yaml", 'r'))


def start(args):
    myenv = world.environment.pusher.PusherEnvGenerator(ENV_CONFIG)

    # setup an optimization
    os.makedirs(args.logs_path, exist_ok=True)
    eta = tf.Variable(args.lr)
    eta_value = tf.keras.optimizers.schedules.ExponentialDecay(args.lr, args.lr_decay_steps, args.lr_decay_rate)
    eta.assign(eta_value(0))
    optimizer = tf.keras.optimizers.Adam(eta)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=myenv.haptic_regressor)
    ckpt_man = tf.train.CheckpointManager(ckpt, args.logs_path, max_to_keep=10)
    train_writer = tf.summary.create_file_writer(args.logs_path + "/train")

    # start training
    metric_loss = tf.keras.metrics.Mean(name="MeanLoss")
    for n_ep in range(NUM_EPISODES):
        for n_act in range(NUM_ACTIONS):
            action = world.action.primitives.PushAction.random_sample()
            observations, reward, done, info = myenv.step(action=action)

            with tf.GradientTape() as tape:
                y_pred = myenv.haptic_regressor(info["observations_numpy"], training=True)

                y_true = np.asarray([v for v in info["haptic"].values()])[np.newaxis, ...]
                loss = tf.keras.losses.mean_squared_error(y_true=y_true, y_pred=y_pred)

                gradients = tape.gradient(loss, myenv.haptic_regressor.trainable_variables)
                optimizer.apply_gradients(zip(gradients, myenv.haptic_regressor.trainable_variables))

            metric_loss.update_state(loss.numpy())

        # do each epoch
        eta.assign(eta_value(0))
        ckpt_man.save()
        myenv.reset()
        myenv.haptic_regressor.reset_states()

        # add to the tensorboard
        print("Episode: {}, mean loss: {}".format(n_ep, metric_loss.result().numpy()))
        with train_writer.as_default():
            tf.summary.scalar(metric_loss.name, metric_loss.result().numpy(), step=n_ep)
            tf.summary.scalar(eta.name, eta.value().numpy(), step=n_ep)
        train_writer.flush()

    myenv.stop_sim()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--logs-path', type=str, default="./logs")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-decay-steps', type=float, default=1000)
    parser.add_argument('--lr-decay-rate', type=float, default=0.99)
    args, _ = parser.parse_known_args()
    world.physics.utils.allow_memory_growth()

    start(args)
