import numpy as np
import tensorflow as tf

from world.action.primitives import PushAction
from world.environment.base import BaseEnv
from world.physics.phys_net import HapticRegressor


class PusherEnvGenerator(BaseEnv):
    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def get_model(config):
        return HapticRegressor(batch_size=config["batch_size"],
                               num_outputs=config["num_outputs"],
                               action_kernel_size=config["action_kernel_size"],
                               dropout=config["dropout"],
                               lstm_units=config["lstm_units"],
                               stateful_lstm=config["lstm_stateful"])

    def step(self, action: PushAction = None):
        assert type(action) is PushAction and self.object is not None
        observations, reward, done, info = list(), None, False, {}

        info["haptic"] = self.rog.get_haptic_values()
        observations = self.get_observations(action)

        obs_flat = observations.reshape((1, -1))
        act_flat = action.to_numpy().reshape((1, -1))
        info["observations_numpy"] = (obs_flat, act_flat)

        return observations, reward, done, info

    def reset(self):
        self.reset_sim()


class RLPusherEnvGenerator(BaseEnv):
    def __init__(self, config):
        super().__init__(config)

        self.done = False
        self.time_penalty = 0

        self.nn = HapticRegressor(batch_size=config["batch_size"],
                                  num_outputs=config["num_outputs"],
                                  action_kernel_size=config["action_kernel_size"],
                                  dropout=config["dropout"],
                                  lstm_units=config["lstm_units"],
                                  stateful_lstm=config["lstm_stateful"])
        path = tf.train.latest_checkpoint(self.config["load_path"])
        ckpt = tf.train.Checkpoint(model=self.nn)
        ckpt.restore(path).expect_partial()
        print(f"Model loaded from {self.config['load_path']}")

    def step(self, action: PushAction = None):
        assert type(action) is PushAction and self.object is not None
        observations, reward, done, info = list(), None, False, {}

        info["haptic"] = self.rog.get_haptic_values()
        observations = self.get_observations(action)

        obs_flat = observations.reshape((1, -1))
        act_flat = action.to_numpy().reshape((1, -1))
        info["observations_numpy"] = (obs_flat, act_flat)

        feed = [f[np.newaxis, ...] for f in info["observations_numpy"]]
        y_pred = self.nn(feed, training=False).numpy()
        y_true = np.asarray([v for v in info["haptic"].values()])
        reward = 1.0 / (1e-6 + np.linalg.norm(y_true - y_pred))
        reward -= self.time_penalty
        self.time_penalty += 0.1

        self.done = bool(reward < 0.1)

        return observations, reward, self.done, info

    def reset(self):
        self.reset_sim()
        self.nn.reset_states()
        self.done = False
        self.time_penalty = 0
