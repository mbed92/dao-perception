import gym
import numpy as np
import pybullet as p
import pybullet_data as pd

from world.environment.objects import RandomObjectsGenerator


class PusherEnv(gym.Env):
    def __init__(self):
        # self.end_pos = config["corridor_length"]
        # self.cur_pos = 0
        # self.action_space = Discrete(2)
        # self.observation_space = Box(0.0, self.end_pos, shape=(1,), dtype=np.float32)
        #
        # # Set the seed. This is only used for the final (reach goal) reward.
        # self.seed(config.worker_index * config.num_workers)

        self.start_sim()
        self.flags = p.URDF_INITIALIZE_SAT_FEATURES
        useFixedBase = True
        self.scene = {
            "plane": p.loadURDF("plane.urdf", [0, 0, -0.5], flags=self.flags, useFixedBase=useFixedBase),
            "pusher": p.loadURDF("cube.urdf", [0, -0.5, 0], flags=self.flags, globalScaling=0.3),
        }

        # generate random scene
        self.objects = {}
        self.rog = RandomObjectsGenerator(0.0, 0.1, 0.0,
                                          0.5, 0.0, 0.0,
                                          rand_z=False, globalScaling_mean=0.3)
        self.randomize_environment()

        # calculate camera position
        self.camTargetPos = [0, 0.1, 0.5]
        self.yaw = 0
        self.pitch = -90.0
        self.roll = 0
        self.upAxisIndex = 2
        self.camDistance = 1.5
        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(self.camTargetPos, self.camDistance,
                                                              self.yaw, self.pitch, self.roll,
                                                              self.upAxisIndex)

        # get projection matrix
        self.pixelWidth = 640
        self.pixelHeight = 480
        self.nearPlane = 0.01
        self.farPlane = 100
        self.fov = 60
        self.aspect = self.pixelWidth / self.pixelHeight
        self.projectionMatrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.nearPlane, self.farPlane)
        p.setRealTimeSimulation(1)

    def reset(self):
        pass
        # self.cur_pos = 0
        # return [self.cur_pos]

    def step(self, action):
        pass
        # assert action in [0, 1], action
        # if action == 0 and self.cur_pos > 0:
        #     self.cur_pos -= 1
        # elif action == 1:
        #     self.cur_pos += 1
        # done = self.cur_pos >= self.end_pos
        # # Produce a random reward when we reach the goal.
        # return [self.cur_pos], \
        #        random.random() * 2 if done else -0.1, done, {}

    def seed(self, seed=None):
        pass
        # random.seed(seed)

    def start_sim(self):
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pd.getDataPath())
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI)
        p.setGravity(0, 0, -9.80991)

    def stop_sim(self):
        p.disconnect()

    def get_camera_image(self, color=True, raw=False):
        img_arr = p.getCameraImage(self.pixelWidth, self.pixelHeight, self.viewMatrix, self.projectionMatrix)
        w, h = img_arr[0], img_arr[1]
        get_data_idx = 2 if color else 3
        num_channels = 4 if color else 1
        img = img_arr[get_data_idx]
        np_img_arr = np.reshape(img, (h, w, num_channels))

        if not raw:
            np_img_arr = np_img_arr * (1. / 255.)

        return np_img_arr

    def randomize_environment(self):
        if self.objects is not None and len(self.objects.keys()) > 0:
            [p.removeBody(obj_id) for obj_id in self.objects.values()]
            self.objects = dict()

        objects = dict()
        for i in range(5):
            key = f"obj_{i}"
            objects[key] = self.rog()
        self.objects = objects

        # get haptic distribution
        ph = ...
        # return ph
