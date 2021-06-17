import gym
import numpy as np
import pybullet as p
import pybullet_data as pd
from ray.rllib.env import EnvContext

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

        # generate random scene
        self.num_objects = 1
        self.objects = {}
        self.rog = RandomObjectsGenerator(mean_y=0.1, sigma_y=0.3, rand_z=False,
                                          additional_objects_folder=config["additional_objects_folder"])

        # calculate camera position
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

    def reset(self):
        # return <obs>

        # reset pusher
        p.resetBasePositionAndOrientation(self.scene["pusher"],
                                          self.config["pusher_position"],
                                          self.config["pusher_quaternion"])

        # reset scene
        if self.objects is not None and len(self.objects.keys()) > 0:
            [p.removeBody(obj_id) for obj_id in self.objects.values()]
            self.objects = dict()

        # set new objects
        objects = dict()
        for i in range(self.num_objects):
            key = f"obj_{i}"
            objects[key] = self.rog()
        self.objects = objects

    def step(self, action=None):
        # return <obs>, <reward: float>, <done: bool>, <info: dict>

        # assert action in [0, 1], action
        # if action == 0 and self.cur_pos > 0:
        #     self.cur_pos -= 1
        # elif action == 1:
        #     self.cur_pos += 1
        # done = self.cur_pos >= self.end_pos
        # # Produce a random reward when we reach the goal.
        # return [self.cur_pos], \
        #        random.random() * 2 if done else -0.1, done, {}
        p.stepSimulation()

    def seed(self, seed=None):
        np.random.seed(seed)

    def start_sim(self):
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pd.getDataPath())
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI)
        p.setGravity(0, 0, -9.80991)

    def stop_sim(self):
        p.disconnect()

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
