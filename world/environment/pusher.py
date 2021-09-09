import cv2
import numpy as np
import pybullet as p
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from utils.text import TextFlag, log
from world.action.primitives import PushAction
from world.environment.base import BaseEnv
from world.physics.phys_net import HapticRegressor


class PusherEnvDemo(BaseEnv):
    def __init__(self, config):
        super().__init__(config)
        self._observations_size = 21

    @staticmethod
    def get_model(config):
        return HapticRegressor(batch_size=config["batch_size"],
                               num_outputs=config["num_outputs"],
                               action_kernel_size=config["action_kernel_size"],
                               dropout=config["dropout"],
                               lstm_units=config["lstm_units"],
                               stateful_lstm=config["lstm_stateful"])

    def get_observations(self, action: PushAction):
        observations = list()

        # set new position of the pusher w. r. t. the object
        if self.scene["pusher"] is not None:
            p.removeBody(self.scene["pusher"])

        state_before = p.getBasePositionAndOrientation(self.object)
        self.scene["pusher"], _, _ = self.setup_pusher(object_pos=state_before[0], action=action)
        observations.append(state_before)

        if action is not None:
            # apply force on a pusher object
            self.step_sim_with_force(action)
            state_after = p.getBasePositionAndOrientation(self.object)
            observations.append(state_after)

            # wait more and get new observation
            self.step_sim_with_force(action)
            state_post_after = p.getBasePositionAndOrientation(self.object)
            observations.append(state_post_after)

        # flatten
        observations = np.asarray([np.hstack(o) for o in observations])
        observations = observations.reshape((1, -1))
        return observations

    def step(self, action: PushAction = None):
        assert type(action) is PushAction and self.object is not None
        observations, reward, done, info = list(), None, False, {}

        # mock observations
        info["haptic"] = self.rog.get_haptic_values()
        observations = self.get_observations(action)
        observations = np.asarray([np.hstack(o) for o in observations])
        obs_flat = observations.reshape((1, -1))
        act_flat = action.to_numpy().reshape((1, -1))
        info["observations_numpy"] = (obs_flat, act_flat)

        return observations, reward, done, info

    def reset(self):
        self.reset_sim()


class RLPusherEnvHapticProperties(py_environment.PyEnvironment, BaseEnv):
    def __init__(self, config):
        py_environment.PyEnvironment.__init__(self)
        BaseEnv.__init__(self, config)

        self._episode_ended = False
        self._time_penalty = config["time_penalty_init"]
        self._time_penalty_delta = config["time_penalty_delta"]
        self._termination_reward = config["termination_reward"]
        self._termination_steps = config["termination_steps"]
        self._steps = 0
        self._observations_size = 21

        # define specs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,),
            dtype=np.float32,
            minimum=np.array([5.0, -np.pi]),
            maximum=np.array([10.0, np.pi]),
            name='push')

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1, self._observations_size),
            dtype=np.float32,
            minimum=-10.0,
            maximum=10.0,
            name='map')

        self._state = np.array([[0.0] * self._observations_size], dtype=np.float32)

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
        log(TextFlag.INFO, f"Model loaded from {self.config['load_path']}")

    def get_observations(self, action: PushAction):
        observations = list()

        # set new position of the pusher w. r. t. the object
        if self.scene["pusher"] is not None:
            p.removeBody(self.scene["pusher"])

        state_before = p.getBasePositionAndOrientation(self.object)
        self.scene["pusher"], _, _ = self.setup_pusher(object_pos=state_before[0], action=action)
        observations.append(state_before)

        if action is not None:
            # apply force on a pusher object
            self.step_sim_with_force(action)
            state_after = p.getBasePositionAndOrientation(self.object)
            observations.append(state_after)

            # wait more and get new observation
            self.step_sim_with_force(action)
            state_post_after = p.getBasePositionAndOrientation(self.object)
            observations.append(state_post_after)

        # flatten
        observations = np.asarray([np.hstack(o) for o in observations])
        observations = observations.reshape((1, -1))
        return observations

    def _reset(self):
        self.reset_sim()

        if not self.nn.stateful_lstm:
            self.nn.reset_states()

        self._steps = 0
        self._episode_ended = False
        self._state = np.array([[0.0] * self._observations_size], dtype=np.float32)
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
        self._observations_size = observations.shape[-1]
        self._state = np.asarray(observations.reshape((1, -1)), dtype=np.float32)

        # calculate a reward
        reward = haptic_reward_with_time_penalty(state=self._state,
                                                 action=action,
                                                 y_true=info["haptic"],
                                                 haptic_regressor=self.nn,
                                                 steps=self._steps,
                                                 time_penalty_delta=self._time_penalty_delta)

        # terminate if needed
        if reward > self._termination_reward or self._steps > self._termination_steps:
            self._episode_ended = True

        # return a result
        self._steps += 1
        if self._episode_ended:
            return ts.termination(self._state, reward=reward)
        else:
            return ts.transition(self._state, reward=reward, discount=1.0)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec


class RLPusherEnvPushNetFromImages(py_environment.PyEnvironment, BaseEnv):
    def __init__(self, config):
        py_environment.PyEnvironment.__init__(self)
        BaseEnv.__init__(self, config)

        self._episode_ended = False
        self._time_penalty = config["time_penalty_init"]
        self._time_penalty_delta = config["time_penalty_delta"]
        self._termination_reward = config["termination_reward"]
        self._termination_steps = config["termination_steps"]
        self._steps = 0
        self._observations_size = 21

        # define specs POSES
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,),
            dtype=np.float32,
            minimum=np.array([5.0, -np.pi]),
            maximum=np.array([10.0, np.pi]),
            name='push')

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1, self._observations_size),
            dtype=np.float32,
            minimum=-10.0,
            maximum=10.0,
            name='map')

        self._state = np.array([[0.0] * self._observations_size], dtype=np.float32)

    def get_masked_depth(self):
        depth_before = self.get_depth_image()
        color_before = self.get_color_image(raw=True)[..., :3]
        mask_before = get_object_mask(color_before,
                                      self.config["object_color_threshold_low"],
                                      self.config["object_color_threshold_high"])
        return depth_before * mask_before

    def place_camera_in_front_of_object(self, cam_dist=0.7, roll=0.0, pitch=-20, yaw=0.0):
        target_point, _ = p.getBasePositionAndOrientation(self.object)
        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(np.asarray(target_point), cam_dist,
                                                              yaw + 90, pitch, roll, 2)

    def get_observations(self, action: PushAction):
        observations = list()

        # set new position of the pusher w. r. t. the object
        if self.scene["pusher"] is not None:
            p.removeBody(self.scene["pusher"])

        state_before = p.getBasePositionAndOrientation(self.object)
        observations.append(state_before)
        self.scene["pusher"], _, pusher_orientation_quaternion = self.setup_pusher(object_pos=state_before[0],
                                                                                   action=action)

        # reposition a camera
        yaw = np.rad2deg(p.getEulerFromQuaternion(pusher_orientation_quaternion)[-1])
        self.place_camera_in_front_of_object(cam_dist=self.config["pusher_radius"] - 0.2, yaw=yaw)

        # get a masked depth image
        masked_depth_before = self.get_masked_depth()
        observations.append(masked_depth_before)

        # apply a force on a pusher object
        if action is not None:
            self.step_sim_with_force(action)

            # clear a view from the camera
            p.removeBody(self.scene["pusher"])
            self.scene["pusher"] = None

            masked_depth_after = self.get_masked_depth()
            observations.append(masked_depth_after)

        return observations

    def _reset(self):
        self.reset_sim()
        self._steps = 0
        self._episode_ended = False
        self._state = np.array([[0.0] * self._observations_size], dtype=np.float32)
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        if type(action) is np.ndarray:
            action = PushAction.from_numpy(action)

        observations, reward, info = list(), None, {}
        info["haptic"] = self.rog.get_haptic_values()
        observations = self.get_observations(action)
        self._observations_size = observations.shape[-1]
        self._state = np.asarray(observations, dtype=np.float32)

        # calculate a reward
        reward = 0.0

        # terminate if needed
        if reward > self._termination_reward or self._steps > self._termination_steps:
            self._episode_ended = True

        # return a result
        self._steps += 1
        if self._episode_ended:
            return ts.termination(self._state, reward=reward)
        else:
            return ts.transition(self._state, reward=reward, discount=1.0)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec


def get_object_mask(img, color_bgr_low, color_bgr_high, k=np.ones((5, 5))):
    assert type(img) is np.ndarray and len(img.shape) == 3 and img.shape[-1] == 3
    assert type(color_bgr_low) in [tuple, list] and len(color_bgr_low) == 3
    assert type(color_bgr_high) in [tuple, list] and len(color_bgr_high) == 3
    assert all([c >= 0] for c in color_bgr_low)
    assert all([c >= 0] for c in color_bgr_high)
    assert all([high > low for low, high in zip(color_bgr_low, color_bgr_high)])

    img_cpy = cv2.medianBlur(img, 3)
    mask = cv2.inRange(img_cpy, np.asarray(color_bgr_low), np.asarray(color_bgr_high))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask[..., np.newaxis]


# REWARD FUNCTIONS
def haptic_reward_with_time_penalty(state, action, **kwargs):
    assert "haptic_regressor" in kwargs.keys()
    assert "y_true" in kwargs.keys()
    assert "steps" in kwargs.keys()
    assert "time_penalty_delta" in kwargs.keys()

    feed = (state, action.to_numpy().reshape((1, -1)))
    feed = [f[np.newaxis, ...] for f in feed]
    y_pred = kwargs["haptic_regressor"](feed, training=False)
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
