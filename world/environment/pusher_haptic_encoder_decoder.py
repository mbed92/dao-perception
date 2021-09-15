import os

import numpy as np
import pybullet as p
import tensorflow as tf
import tensorflow_addons as tfa
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from utils.text import TextFlag, log
from world.action.primitives import PushAction
from world.environment.base import BaseEnv
from world.environment.utils import get_object_mask, place_camera_in_front_of_object
from world.physics.haptic_encoder_decoder import HapticEncoderDecoder


def depth_images_loss(y_true, y_hat):
    mask_true = tf.cast(tf.clip_by_value(y_true, 0, 1), tf.bool)
    mask_pred = tf.cast(tf.clip_by_value(y_hat, 0, 1), tf.bool)
    # iou_loss = tfa.losses.giou_loss(mask_true, mask_pred)
    abs_loss = tf.reduce_mean(tf.keras.losses.mean_absolute_error(y_true, y_hat))
    return abs_loss


class PushNetEncoderDecoder(py_environment.PyEnvironment, BaseEnv):
    def __init__(self, config):
        py_environment.PyEnvironment.__init__(self)
        BaseEnv.__init__(self, config)

        self._episode_ended = False
        self._time_discount = config["time_discount"]
        self._termination_reward = config["termination_reward"]
        self._termination_steps = config["termination_steps"]
        self._steps = 0
        self._global_steps_num = 0
        self._observations_size = [2, 480, 640, 1]

        # define specs POSES
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,),
            dtype=np.float32,
            minimum=np.array([-np.pi, 5.0]),
            maximum=np.array([np.pi, 20.0]),
            name='push')

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=self._observations_size,
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name='map')

        self._state = np.zeros(shape=self._observations_size, dtype=np.float32)
        self._gradients = list()

        # setup a haptic network
        self._predictive_model, \
        self._eta, \
        self._eta_value, \
        self._haptic_optimizer, \
        self._haptic_ckpt_man, \
        self._haptic_metric_loss, \
        self._haptic_writer = self._setup_predictive_model()

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
        self.viewMatrix = place_camera_in_front_of_object(self.object, self.config["pusher_radius"] - 0.2, yaw=yaw)

        # get a masked depth image
        depth_before = self.get_depth_image()
        color_before = self.get_color_image(raw=True)[..., :3]
        mask_before = get_object_mask(color_before, self.config["object_color_threshold_low"],
                                      self.config["object_color_threshold_high"])
        masked_depth_before = depth_before * mask_before
        observations.append(masked_depth_before)

        # apply a force on a pusher object
        self.step_sim_with_force(action)

        # clear a view from the camera
        p.removeBody(self.scene["pusher"])
        self.scene["pusher"] = None

        # get depth image after an action
        depth_after = self.get_depth_image()
        color_after = self.get_color_image(raw=True)[..., :3]
        mask_after = get_object_mask(color_after, self.config["object_color_threshold_low"],
                                     self.config["object_color_threshold_high"])
        masked_depth_after = depth_after * mask_after
        observations.append(masked_depth_after)

        return np.asarray(observations)

    def _optimize_haptic_net(self):
        self._haptic_optimizer.apply_gradients(zip(self._gradients, self._predictive_model.trainable_variables))
        self._global_steps_num += 1
        log(TextFlag.WARNING, f"Haptic Net optimized! Num optimization steps: {self._global_steps_num}")

    def _reset(self):
        self.reset_sim()
        self._steps = 0
        self._episode_ended = False
        self._state = np.zeros(shape=self._observations_size, dtype=np.float32)

        if len(self._gradients) > 0:
            self._optimize_haptic_net()
        self._gradients = list()

        # add haptic loss to the tensorboard
        with self._haptic_writer.as_default():
            tf.summary.scalar(self._haptic_metric_loss.name, self._haptic_metric_loss.result().numpy(),
                              step=self._global_steps_num)
        self._haptic_writer.flush()

        if self._global_steps_num and self._global_steps_num % self.config["haptic_net_save_period"] == 0:
            self._eta.assign(self._eta_value(self._global_steps_num))
            self._haptic_ckpt_man.save()
            path = self.config["save_path"]
            log(TextFlag.WARNING, f"Haptic net saved at {path}")

        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        if tf.is_tensor(action):
            action = action.numpy()
        if type(action) is np.ndarray:
            action = PushAction.from_numpy(action)

        observations, reward, info = list(), None, {}
        info["haptic"] = self.rog.get_haptic_values()
        observations = self.get_observations(action)
        self._observations_size = observations.shape

        if self.config["DEBUG"]:
            log(TextFlag.INFO,
                f"step: {self._steps}\taction: {action}\t observations: {observations.shape}\t"
                f"isNaN: {np.isnan(observations).any()}\tisInf: {np.isinf(observations).any()}\t")

        if not (np.isnan(observations).any() or np.isinf(observations).any()):
            self._state = np.asarray(observations, dtype=np.float32)
        else:
            log(TextFlag.WARNING, "state is NAN of INF!")
            self._reset()

        # gather experience
        depth_before = observations[0][np.newaxis, ...]
        depth_after = observations[1][np.newaxis, ...]
        with tf.GradientTape() as tape:
            predicted_depth_hat, metrics = self._predictive_model([depth_before, action.to_numpy()], training=True)
            predicted_depth_true = tf.convert_to_tensor(depth_after)
            loss_no_reg = depth_images_loss(predicted_depth_true, predicted_depth_hat)
            l2_reg = tf.add_n(
                [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in self._predictive_model.trainable_variables]
            ) * 0.001
            loss_reg = loss_no_reg + l2_reg

        # add loss to tensorboard
        self._haptic_metric_loss.update_state(loss_no_reg.numpy())

        # get mean gradients
        gradients = tape.gradient(loss_reg, self._predictive_model.trainable_variables)
        if len(self._gradients) > 1 and len(self._gradients) == len(gradients):
            for i in range(len(self._gradients)):
                self._gradients[i] = (self._gradients[i] + gradients[i]) / 2.0
        else:
            self._gradients = list(gradients)
        self._gradients.append(gradients)

        # calculate a reward
        reward = 0.0

        # return a result
        self._steps += 1
        if self._episode_ended:
            return ts.termination(self._state, reward=reward)
        else:
            return ts.transition(self._state, reward=reward, discount=self._time_discount)

    def _setup_predictive_model(self):
        model = HapticEncoderDecoder(batch_size=self.config["batch_size"],
                                     dropout=self.config["dropout"])

        # initialize a model
        mock_action = PushAction.random_sample(10).to_numpy()
        state_before_action = self._state[0][np.newaxis, ...]
        model([state_before_action, mock_action], training=True)

        # learning rate
        eta = tf.Variable(float(self.config["lr"]))
        eta_value = tf.keras.optimizers.schedules.ExponentialDecay(float(self.config["lr"]),
                                                                   float(self.config["lr_decay_steps"]),
                                                                   float(self.config["lr_decay_rate"]))
        eta.assign(eta_value(0))

        # create optimizer
        optimizer = tf.keras.optimizers.Adam(eta)
        os.makedirs(self.config["save_path"], exist_ok=True)
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
        ckpt_man = tf.train.CheckpointManager(ckpt, self.config["save_path"], max_to_keep=10)

        # add a metric writer
        metric_loss = tf.keras.metrics.Mean(name="MeanLoss")
        writer = tf.summary.create_file_writer(os.path.join(self.config["save_path"], "haptic_net_train"))

        return model, eta, eta_value, optimizer, ckpt_man, metric_loss, writer
