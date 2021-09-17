from argparse import ArgumentParser

import tensorflow as tf
import yaml

import world

tf.executing_eagerly()
ENV_CONFIG = yaml.safe_load(open("../config/train_haptic_rl_random.yaml", 'r'))


def start(args):
    env = world.environment.pusher_haptic_encoder_decoder.PushNetEncoderDecoder(ENV_CONFIG)
    env.reset()

    # start training
    for n_ep in range(args.n_episodes_train):
        for n_act in range(args.n_actions):
            action = world.action.primitives.PushAction.random_sample()
            env.step(action)

        # update weights and restart env
        env.reset()

    env.stop_sim()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--n-episodes-train', type=int, default=10000)
    parser.add_argument('--n-actions', type=int, default=10)
    args, _ = parser.parse_known_args()
    world.physics.utils.allow_memory_growth()
    start(args)
