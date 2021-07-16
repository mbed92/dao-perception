import os
from argparse import ArgumentParser

import tensorflow as tf
import yaml

import utils
import world

tf.executing_eagerly()
ENV_CONFIG = yaml.safe_load(open("../config/train_haptic_offline.yaml", 'r'))


def start(args):
    model = world.environment.pusher.PusherEnvGenerator(ENV_CONFIG).get_model(ENV_CONFIG)

    # load datasets
    train_ds, val_ds = utils.dataset.load(args.dataset_train, batch_size=args.batch_size, split_ratio=args.split_train_ratio)
    test_ds = utils.dataset.load(args.dataset_train, batch_size=args.batch_size)

    # setup an optimization
    os.makedirs(args.logs_path, exist_ok=True)
    eta = tf.Variable(args.lr)
    eta_value = tf.keras.optimizers.schedules.ExponentialDecay(args.lr, args.lr_decay_steps, args.lr_decay_rate)
    eta.assign(eta_value(0))
    optimizer = tf.keras.optimizers.Adam(eta)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_man = tf.train.CheckpointManager(ckpt, args.logs_path, max_to_keep=10)
    train_writer, train_step = tf.summary.create_file_writer(args.logs_path + "/train"), 0
    val_writer, val_step = tf.summary.create_file_writer(args.logs_path + "/val"), 0
    test_writer, test_step = tf.summary.create_file_writer(args.logs_path + "/test"), 0

    # start training
    for n_ep in range(args.epochs):
        train_step = utils.epoch.run(model, train_ds, train_writer, train_step, "train_", optimizer, eta)
        val_step = utils.epoch.run(model, val_ds, val_writer, val_step, "validation_")
        test_step = utils.epoch.run(model, test_ds, test_writer, test_step, "test_")
        eta.assign(eta_value(n_ep))
        ckpt_man.save()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--logs-path', type=str, default="./logs_offline")
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-decay-steps', type=float, default=200)
    parser.add_argument('--lr-decay-rate', type=float, default=0.99)
    parser.add_argument('--split-train-ratio', type=float, default=0.9)
    parser.add_argument('--dataset-train', type=str,
                        default="/media/mbed/internal/backup/rl-physnet/train5000_test500x10/train_1626436481.npy")
    parser.add_argument('--dataset-test', type=str,
                        default="/media/mbed/internal/backup/rl-physnet/train5000_test500x10/test_1626436481.npy")

    args, _ = parser.parse_known_args()
    world.physics.utils.allow_memory_growth()

    start(args)
