import time

import gym
import numpy as np
import pybullet as p
import pybullet_data as pd
from ray.rllib.env import EnvContext
from scipy.spatial.transform import Rotation as R

from world.environment.objects import RandomObjectsGenerator

GRAVITY = -9.80991


def pose_on_circle(radius, yaw, height, pos_offset=None):
    x = np.cos(yaw) * radius
    y = np.sin(yaw) * radius
    pos = np.asarray([x, y, height])

    if pos_offset is not None:
        pos += np.asarray(pos_offset)

    quat = R.from_euler('z', yaw).as_quat()
    return pos, quat


class BaseEnv(gym.Env):
    def __init__(self, config: EnvContext):
        self.config = config
        self.flags = p.RESET_USE_DEFORMABLE_WORLD
        self.scene = {}
        self.rog = RandomObjectsGenerator(self.config["object_position"], [0, 0, 0, 1],
                                          self.config["object_size_mean"], self.config["object_size_sigma"],
                                          self.config["object_mass_mean"], self.config["object_mass_sigma"],
                                          self.config["object_friction_mean"], self.config["object_friction_sigma"],
                                          self.config["object_restitution_mean"],
                                          self.config["object_restitution_sigma"],
                                          self.config["object_spring_stiffness_mean"],
                                          self.config["object_spring_stiffness_sigma"],
                                          self.config["object_elastic_stiffness_mean"],
                                          self.config["object_elastic_stiffness_sigma"])

        self.object = None

        # start the simulation
        self.start_sim()

        # calculate camera position
        self.setup_camera()

    def seed(self, seed=None):
        np.random.seed(seed)

    def get_observations(self, action):
        observations = list()

        # set new position of the pusher
        p.removeBody(self.scene["pusher"])
        object_pos, _ = p.getBasePositionAndOrientation(self.object)
        self.scene["pusher"] = self.setup_pusher(object_pos=object_pos, action=action)
        observations.append(object_pos)

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
            p.setJointMotorControl2(self.scene["pusher"], 1, p.POSITION_CONTROL, targetPosition=-10,
                                    force=action.force, maxVelocity=self.config["pusher_lin_vel"])
            p.stepSimulation()
            time.sleep(1.0 / 240.)

    def start_sim(self):
        if self.config["simulation_use_gui"]:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.reset_sim()

    def reset_sim(self):
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setGravity(0, 0, GRAVITY)

        self.scene["plane"] = self.setup_scene()
        self.scene["pusher"] = self.setup_pusher()

        try:
            self.object = self.rog.generate_object()
        except ValueError as e:
            print(e)

    def stop_sim(self):
        p.disconnect()

    def setup_scene(self):
        plane_id = p.loadURDF("plane.urdf", self.config["plane_position"], self.config["plane_quaternion"])
        p.changeDynamics(bodyUniqueId=plane_id, linkIndex=-1, mass=0, restitution=1.0, lateralFriction=1.0)
        return plane_id

    def setup_pusher(self, object_pos=None, action=None):
        # create collision boxes
        base = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
        link = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.02, 0.02])
        pusher = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.05, 0.1])

        # pusher position around the object on a circle
        pos_offset = self.config["object_position"]
        if object_pos is not None and action is not None:
            yaw = action.yaw
            pos_offset += np.array(object_pos)
        else:
            yaw = 0.0

        base_position, base_orientation = pose_on_circle(radius=self.config["pusher_radius"],
                                                         yaw=yaw,
                                                         height=self.config["pusher_height"],
                                                         pos_offset=pos_offset)

        baseMass = 0  # fixed
        baseCollisionShapeIndex = base
        baseVisualShapeIndex = -1
        linkMasses = [0.5, 0.01, 0.01]
        linkCollisionShapeIndices = [-1, link, pusher]
        linkVisualShapeIndices = [-1, link, pusher]
        linkPositions = [[0, 0, -0.1], [-0.1, 0, 0], [-0.1, 0, 0]]
        linkOrientations = [[0, 0, 0, 1], [0, -0.0663219, 0, 0.9977983], [0, 0, 0, 1]]
        linkInertialFramePositions = linkPositions
        linkInertialFrameOrientations = linkOrientations
        linkParentIndices = [0, 1, 2]
        linkJointTypes = [p.JOINT_FIXED, p.JOINT_PRISMATIC, p.JOINT_REVOLUTE]
        linkJointAxis = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]

        pusher_id = p.createMultiBody(baseMass=baseMass,
                                      baseCollisionShapeIndex=baseCollisionShapeIndex,
                                      baseVisualShapeIndex=baseVisualShapeIndex,
                                      basePosition=base_position,
                                      baseOrientation=base_orientation,
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
        p.createConstraint(parentBodyUniqueId=self.scene["plane"],
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
