import time

import gym
import numpy as np
import pybullet as p
from ray.rllib.env import EnvContext

from world.environment.objects import RandomObjectsGenerator


class BaseEnv(gym.Env):
    def __init__(self, config: EnvContext):
        self.config = config
        self.start_sim()
        self.flags = p.URDF_INITIALIZE_SAT_FEATURES
        self.pivot = None
        self.scene = {
            "plane": self.setup_scene(),
            "pusher": self.setup_pusher()
        }

        # generate random object in the scene
        self.rog = RandomObjectsGenerator(self.config["object_position_mean"], self.config["object_position_sigma"],
                                          self.config["object_quaternion"],
                                          self.config["object_size_mean"], self.config["object_size_sigma"],
                                          self.config["object_mass_mean"], self.config["object_mass_sigma"],
                                          self.config["object_friction_mean"], self.config["object_friction_sigma"])
        self.object = None

        # calculate camera position
        self.setup_camera()

    def seed(self, seed=None):
        np.random.seed(seed)

    def get_observations(self, action):
        observations = list()

        # get current state
        state_before = p.getBasePositionAndOrientation(self.object)
        observations.append(state_before)

        if action is not None:
            # apply force on a pusher object
            self.step_sim_with_force(self.config["simulation_action_steps"], action)
            state_after = p.getBasePositionAndOrientation(self.object)
            observations.append(state_after)

            # wait more and get new observation
            self.step_sim_with_force(self.config["simulation_action_steps"], action)
            state_post_after = p.getBasePositionAndOrientation(self.object)
            observations.append(state_post_after)

        return observations

    def step_sim_with_force(self, action_steps, action):
        t_end = time.time() + action_steps * self.config["simulation_timestep"]

        while time.time() < t_end:
            p_pos, _ = p.getBasePositionAndOrientation(self.scene["pusher"])
            p.setJointMotorControl2(self.scene["pusher"], 0, p.POSITION_CONTROL, targetPosition=action.yaw,
                                    force=self.config["pusher_yaw_gain"], maxVelocity=self.config["pusher_yaw_vel"])
            p.setJointMotorControl2(self.scene["pusher"], 1, p.POSITION_CONTROL, targetPosition=0.8,
                                    force=action.force, maxVelocity=self.config["pusher_lin_vel"])
            p.stepSimulation()
            time.sleep(1. / 240.)

    def start_sim(self):
        if self.config["simulation_use_gui"]:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        if type(self.config["objects_path"]) is str:
            p.setAdditionalSearchPath(self.config["objects_path"])

        p.resetSimulation()
        p.setGravity(0, 0, -9.80991)

    def stop_sim(self):
        p.disconnect()

    def reset_sim(self):
        if self.scene["pusher"] is not None:
            p.removeBody(self.scene["pusher"])
        self.scene["pusher"] = self.setup_pusher()

        if self.object is not None:
            p.removeBody(self.object)

        try:
            self.object = self.rog.generate_object()
        except ValueError as e:
            print(e)

    def setup_scene(self):
        p.createCollisionShape(p.GEOM_PLANE)
        return p.createMultiBody(0, 0)

    def setup_pusher(self):
        # create collision boxes
        plane = p.createMultiBody(0, 0)
        base = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
        link = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.15, 0.02, 0.02])
        pusher = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.05, 0.1])

        baseMass = 0
        baseCollisionShapeIndex = base
        baseVisualShapeIndex = -1
        basePosition = self.config["pusher_position"]
        baseOrientation = self.config["plane_quaternion"]
        linkMasses = [0.5, 0.01, 0.01]
        linkCollisionShapeIndices = [-1, link, pusher]
        linkVisualShapeIndices = [-1, link, pusher]
        linkPositions = [[-0.1, 0, -0.13], [0.2, 0, 0], [0.3, 0, 0]]
        linkOrientations = [[0, 0, 0, 1], [0, 0.1305262, 0, 0.9914449], [0, 0, 0, 1]]
        linkInertialFramePositions = [[0, 0, 0]] * len(linkMasses)
        linkInertialFrameOrientations = [[0, 0, 0, 1]] * len(linkMasses)
        linkParentIndices = [0, 1, 2]
        linkJointTypes = [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC, p.JOINT_REVOLUTE]
        linkJointAxis = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]

        pusher_id = p.createMultiBody(baseMass=baseMass,
                                      baseCollisionShapeIndex=baseCollisionShapeIndex,
                                      baseVisualShapeIndex=baseVisualShapeIndex,
                                      basePosition=basePosition,
                                      baseOrientation=baseOrientation,
                                      linkMasses=linkMasses,
                                      linkCollisionShapeIndices=linkCollisionShapeIndices,
                                      linkVisualShapeIndices=linkVisualShapeIndices,
                                      linkPositions=linkPositions,
                                      linkOrientations=linkOrientations,
                                      linkInertialFramePositions=linkInertialFramePositions,
                                      linkInertialFrameOrientations=linkInertialFrameOrientations,
                                      linkParentIndices=linkParentIndices,
                                      linkJointTypes=linkJointTypes,
                                      linkJointAxis=linkJointAxis)

        # make spherical joint compliant
        p.changeDynamics(pusher_id, 2, linearDamping=1e-5, angularDamping=1e-5, jointDamping=1e-5)

        # attach body to the fixed position
        p.createConstraint(parentBodyUniqueId=plane,
                           parentLinkIndex=-1,
                           childBodyUniqueId=pusher_id,
                           childLinkIndex=-1,
                           jointType=p.JOINT_FIXED,
                           jointAxis=[0, 0, 1],
                           parentFramePosition=[0, 0, 1],
                           childFramePosition=[0, 0, 0],
                           parentFrameOrientation=[0, 0, 0, 1],
                           childFrameOrientation=[0, 0, 0, 1]
                           )

        return pusher_id

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
