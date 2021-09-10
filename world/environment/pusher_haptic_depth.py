import os

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
from world.environment.rewards import reward_train_predictive_model
from world.physics.phys_net import CNNClassifier


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


class PusherHapticWithDepth(py_environment.PyEnvironment, BaseEnv):
    def __init__(self, config):
        py_environment.PyEnvironment.__init__(self)
        BaseEnv.__init__(self, config)

        self._episode_ended = False
        self._time_discount = config["time_discount"]
        self._termination_reward = config["termination_reward"]
        self._termination_steps = config["termination_steps"]
        self._steps = 0
        self._observations_size = [2, 480, 640, 1]

        # define specs POSES
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,),
            dtype=np.float32,
            minimum=np.array([-np.pi, 5.0]),
            maximum=np.array([np.pi, 15.0]),
            name='push')

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=self._observations_size,
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name='map')

        self._state = np.zeros(shape=self._observations_size, dtype=np.float32)
        self.predictive_model, self.eta, self.eta_value, self.optimizer, self.ckpt_man = self._setup_predictive_model()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def get_observations(self, action: PushAction):
        observations = list()

        # set new position of the pusher w. r. t. the object
        if self.scene["pusher"] is not None:
            p.removeBody(self.scene["pusher"])

        state_before = p.getBasePositionAndOrientation(self.object)
        self.scene["pusher"], _, pusher_orientation_quaternion = self.setup_pusher(object_pos=state_before[0],
                                                                                   action=action)

        # reposition a camera
        yaw = np.rad2deg(p.getEulerFromQuaternion(pusher_orientation_quaternion)[-1])
        self._place_camera_in_front_of_object(cam_dist=self.config["pusher_radius"] - 0.2, yaw=yaw)

        # get a masked depth image
        masked_depth_before = self._get_masked_depth()
        observations.append(masked_depth_before)

        # apply a force on a pusher object
        if action is not None:
            self.step_sim_with_force(action)

            # clear a view from the camera
            p.removeBody(self.scene["pusher"])
            self.scene["pusher"] = None

            masked_depth_after = self._get_masked_depth()
            observations.append(masked_depth_after)

        # normalize observations
        v_min, v_max = self._observation_spec.minimum, self._observation_spec.maximum
        observations = [np.clip((o - np.mean(o)) / np.std(o), v_min, v_max) for o in observations]
        return np.asarray(observations)

    def _reset(self):
        self.reset_sim()
        self._steps = 0
        self._episode_ended = False
        self._state = np.zeros(shape=self._observations_size, dtype=np.float32)
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        if type(action) is np.ndarray:
            action = PushAction.from_numpy(action)

        observations, reward, info = list(), None, {}
        info["haptic"] = self.rog.get_haptic_values()
        observations = self.get_observations(action)
        self._observations_size = observations.shape
        self._state = np.asarray(observations, dtype=np.float32)

        # calculate a reward
        reward = 0.0
        # reward = reward_train_predictive_model(state=self._state,
        #                                        action=action.to_numpy(),
        #                                        model=self.predictive_model,
        #                                        y_true=info["haptic"],
        #                                        eta=self.eta,
        #                                        eta_value=self.eta_value,
        #                                        optimizer=self.optimizer,
        #                                        ckpt_man=self.ckpt_man)

        # terminate if needed
        if reward > self._termination_reward or self._steps > self._termination_steps:
            self._episode_ended = True

        if self.config["DEBUG"]:
            log(TextFlag.INFO, f"reward: {reward}\tstep: {self._steps}\taction: {action}\t state: {self._state.shape}\t"
                               f"isNaN: {np.isnan(self._state).any()}\tisInf: {np.isinf(self._state).any()}\t")

        # return a result
        self._steps += 1
        if self._episode_ended:
            return ts.termination(self._state, reward=reward)
        else:
            return ts.transition(self._state, reward=reward, discount=self._time_discount)

    def _setup_predictive_model(self):
        model = CNNClassifier(batch_size=self.config["batch_size"],
                              num_outputs=self.config["num_outputs"],
                              action_kernel_size=self.config["action_kernel_size"],
                              dropout=self.config["dropout"],
                              lstm_units=self.config["lstm_units"],
                              stateful_lstm=self.config["lstm_stateful"])

        eta = tf.Variable(float(self.config["lr"]))
        eta_value = tf.keras.optimizers.schedules.ExponentialDecay(float(self.config["lr"]),
                                                                   float(self.config["lr_decay_steps"]),
                                                                   float(self.config["lr_decay_rate"]))
        eta.assign(eta_value(0))

        optimizer = tf.keras.optimizers.Adam(eta)

        os.makedirs(self.config["save_path"], exist_ok=True)
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
        ckpt_man = tf.train.CheckpointManager(ckpt, self.config["save_path"], max_to_keep=10)

        return model, eta, eta_value, optimizer, ckpt_man

    def _get_masked_depth(self):
        depth_before = self.get_depth_image()
        color_before = self.get_color_image(raw=True)[..., :3]
        mask_before = get_object_mask(color_before,
                                      self.config["object_color_threshold_low"],
                                      self.config["object_color_threshold_high"])
        return depth_before * mask_before

    def _place_camera_in_front_of_object(self, cam_dist=0.7, roll=0.0, pitch=-20, yaw=0.0):
        target_point, _ = p.getBasePositionAndOrientation(self.object)
        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(np.asarray(target_point), cam_dist,
                                                              yaw + 90, pitch, roll, 2)
