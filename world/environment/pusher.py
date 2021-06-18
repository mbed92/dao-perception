import gym
import numpy as np
import pybullet as p
import pybullet_data as pd
from ray.rllib.env import EnvContext

from world.action.primitives import PushAction
from world.environment.objects import RandomObjectsGenerator


class PusherEnv(gym.Env):
    def __init__(self, config: EnvContext):
        self.config = config
        self.start_sim()
        self.flags = p.URDF_INITIALIZE_SAT_FEATURES
        self.scene = {
            "plane": p.loadURDF("plane.urdf",
                                self.config["plane_position"], self.config["plane_quaternion"],
                                flags=self.flags, useFixedBase=True),
            "pusher": p.loadURDF("cube.urdf",
                                 self.config["pusher_position"],
                                 self.config["pusher_quaternion"],
                                 globalScaling=self.config["pusher_global_scaling"],
                                 flags=self.flags)
        }

        # generate random object in the scene
        self.rog = RandomObjectsGenerator(mean_y=0.1, sigma_y=0.3, rand_z=False,
                                          additional_objects_folder=config["additional_objects_folder"])
        self.object = self.rog()

        # calculate camera position
        self.setup_camera()

    def reset(self):
        # return <obs>

        # reset pusher
        p.resetBasePositionAndOrientation(self.scene["pusher"],
                                          self.config["pusher_position"],
                                          self.config["pusher_quaternion"])

        # reset scene
        if self.object is not None:
            p.removeBody(self.object)
            self.object = None

        # set new object
        self.object = self.rog()

    def step(self, action: PushAction = None):
        assert type(action) is PushAction
        observations, reward, done, info = list(), None, False, {}

        # get current state
        state_before = p.getBasePositionAndOrientation(self.object)
        observations.append(state_before)

        # execute action
        if action is not None:
            p_pos, _ = p.getBasePositionAndOrientation(self.scene["pusher"])
            p.applyExternalForce(objectUniqueId=self.scene["pusher"], linkIndex=-1,
                                 forceObj=action.push_vector, posObj=p_pos, flags=p.WORLD_FRAME)

            # get current state after
            self.proceed_sim(action.action_time)
            state_after = p.getBasePositionAndOrientation(self.object)
            observations.append(state_after)

            # get current state again
            self.proceed_sim(action.action_time)
            state_post_after = p.getBasePositionAndOrientation(self.object)
            observations.append(state_post_after)

        return observations, reward, done, info

    def seed(self, seed=None):
        np.random.seed(seed)

    def proceed_sim(self, time_to_proceed=1):
        i = 0.0
        while i < time_to_proceed:
            p.stepSimulation()
            i += self.config["simulation_timestep"]

    def start_sim(self):
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pd.getDataPath())
        p.resetSimulation()
        p.setTimeStep(self.config["simulation_timestep"])
        p.setGravity(0, 0, -9.80991)

    def stop_sim(self):
        p.disconnect()

    def setup_camera(self):
        up_axis_idx = 2
        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(self.config["cam_target_pos"],
                                                              self.config["cam_distance"],
                                                              self.config["cam_roll"],
                                                              self.config["cam_pitch"],
                                                              self.config["cam_yaw"],
                                                              up_axis_idx)

        # get projection matrix
        self.aspect = self.config["projection_w"] / self.config["projection_h"]
        self.projectionMatrix = p.computeProjectionMatrixFOV(self.config["fov"],
                                                             self.aspect,
                                                             self.config["near_plane"],
                                                             self.config["far_plane"])

    def get_camera_image(self, color=True, raw=False):
        img_arr = p.getCameraImage(self.config["projection_w"],
                                   self.config["projection_h"],
                                   self.viewMatrix,
                                   self.projectionMatrix)

        w, h = img_arr[0], img_arr[1]
        get_data_idx = 2 if color else 3
        num_channels = 4 if color else 1
        img = img_arr[get_data_idx]
        np_img_arr = np.reshape(img, (h, w, num_channels))

        if not raw:
            np_img_arr = np_img_arr * (1. / 255.)

        return np_img_arr
