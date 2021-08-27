import os
import time
from argparse import ArgumentParser

import numpy as np
import yaml
from numpy import asarray, concatenate

import world
from utils.text import TextFlag, log

ENV_CONFIG = yaml.safe_load(open("../config/gather_data.yaml", 'r'))


def create_dataset(myenv, file, n_episodes, n_actions):
    dataset = {"observations": list(), "actions": list(), "y": list()}
    for n_ep in range(n_episodes):
        batch_obs, batch_act, batch_y = list(), list(), list()
        for n_act in range(n_actions):
            action = world.action.primitives.PushAction.random_sample()
            observations, _, _, info = myenv.step(action=action)
            batch_obs.append(info["observations_numpy"][0])
            batch_act.append(info["observations_numpy"][1])
            batch_y.append(asarray([v for v in info["haptic"].values()]))

        # dump data after each episode and restart an environment
        log(TextFlag.INFO, "Finished episode {}".format(n_ep))
        dataset['observations'].append(concatenate(batch_obs, 0))
        dataset['actions'].append(concatenate(batch_act, 0))
        dataset['y'].append(asarray(batch_y))
        myenv.reset()
    np.save(file, dataset)


def start(args):
    myenv = world.environment.pusher.PusherEnvGenerator(ENV_CONFIG)

    # create folder for data
    os.makedirs(args.data_path, exist_ok=True)

    # save data for test and train
    mytime = int(time.time())
    train_file = os.path.join(args.data_path, "{}_{}.npy".format("train", mytime))
    test_file = os.path.join(args.data_path, "{}_{}.npy".format("test", mytime))

    # generate train dataset with a cube shape
    myenv.rog.object_types = ['cube.obj']
    myenv.reset()
    with open(train_file, 'wb') as ftrain:
        create_dataset(myenv, ftrain, args.n_episodes_train, args.n_actions)

    # generate a test dataset with a different object shape
    myenv.rog.object_types = ['stone_2.obj']
    myenv.reset()
    with open(test_file, 'wb') as ftest:
        create_dataset(myenv, ftest, args.n_episodes_test, args.n_actions)

    myenv.stop_sim()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--data-path', type=str,
                        default="/media/mbed/internal/backup/rl-physnet/train5000_test500x10")
    parser.add_argument('--data-file', type=str, default="data")
    parser.add_argument('--n-episodes-train', type=int, default=5000)
    parser.add_argument('--n-episodes-test', type=int, default=500)
    parser.add_argument('--n-actions', type=int, default=10)
    args, _ = parser.parse_known_args()
    world.physics.utils.allow_memory_growth()

    start(args)
