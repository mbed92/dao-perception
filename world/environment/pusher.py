import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

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
        observations = np.asarray([np.hstack(o) for o in observations])

        obs_flat = observations.reshape((1, -1))
        act_flat = action.to_numpy().reshape((1, -1))
        info["observations_numpy"] = (obs_flat, act_flat)

        return observations, reward, done, info

    def reset(self):
        self.reset_sim()


class RLPusherEnvGenerator(py_environment.PyEnvironment, BaseEnv):
    def __init__(self, config):
        py_environment.PyEnvironment.__init__(self)
        BaseEnv.__init__(self, config)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,),
            dtype=np.float32,
            minimum=np.array([5.0, -np.pi]),
            maximum=np.array([10.0, np.pi]),
            name='push')

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1, self.observations_size),
            dtype=np.float32,
            minimum=-10.0,
            maximum=10.0,
            name='map')

        self._state = np.array([[0.0] * self.observations_size], dtype=np.float32)
        self._episode_ended = False
        self._time_penalty = config["time_penalty_init"]
        self._time_penalty_delta = config["time_penalty_delta"]
        self._termination_reward = config["termination_reward"]
        self._termination_steps = config["termination_steps"]
        self._steps = 0

        # load a haptic net
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

    def _reset(self):
        self.reset_sim()

        if not self.nn.stateful_lstm:
            self.nn.reset_states()

        self._steps = 0
        self._episode_ended = False
        self._state = np.array([[0.0] * self.observations_size], dtype=np.float32)
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        if type(action) is np.ndarray:
            action = PushAction.from_numpy(action)

        observations, reward, info = list(), None, {}
        info["haptic"] = self.rog.get_haptic_values()
        observations = self.get_observations(action)
        observations = np.asarray([np.hstack(o) for o in observations])
        obs_flat = observations.reshape((1, -1))
        self._state = np.asarray(obs_flat, dtype=np.float32)

        # check the object
        info["haptic_regressor_feed"] = (obs_flat, action.to_numpy().reshape((1, -1)))
        feed = [f[np.newaxis, ...] for f in info["haptic_regressor_feed"]]
        y_pred = self.nn(feed, training=False)
        y_pred = tf.reshape(y_pred, -1)

        # calculate a reward
        y_true = tf.convert_to_tensor([v for v in info["haptic"].values()])
        loss = tf.losses.mean_absolute_error(y_true, y_pred)
        reward = 1.0 / (1e-6 + loss)
        reward = reward - self._steps * self._time_penalty_delta
        if reward > self._termination_reward or self._steps > self._termination_steps:
            self._episode_ended = True

        self._steps += 1
        if self._episode_ended:
            return ts.termination(self._state, reward=reward)
        else:
            return ts.transition(self._state, reward=reward, discount=1.0)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec
