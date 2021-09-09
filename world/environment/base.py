import time
from functools import wraps

import numpy as np
import pybullet as p
import pybullet_data as pd
from scipy.spatial.transform import Rotation as R

from utils.text import TextFlag, log
from world.action.primitives import PushAction
from world.environment.objects import RandomObjectsGenerator

GRAVITY = -9.80991


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        log(TextFlag.INFO, f'Func: {f.__name__} took: {te - ts} sec')
        return result

    return wrap


def pose_on_circle(radius, yaw, height, pos_offset=None):
    x = np.cos(yaw) * radius
    y = np.sin(yaw) * radius
    pos = np.asarray([x, y, height])

    if pos_offset is not None:
        pos += np.asarray(pos_offset)

    quat = R.from_euler('z', yaw).as_quat()
    return pos, quat


class BaseEnv:
    def __init__(self, config):
        self.config = config
        self.flags = p.RESET_USE_DEFORMABLE_WORLD
        self.scene = {}
        self.rog = RandomObjectsGenerator(self.config["object_position"], self.config["object_quaternion"],
                                          self.config["object_size_mean"], self.config["object_size_sigma"],
                                          self.config["object_mass_mean"], self.config["object_mass_sigma"],
                                          self.config["object_friction_mean"], self.config["object_friction_sigma"],
                                          self.config["object_restitution_mean"],
                                          self.config["object_restitution_sigma"],
                                          self.config["object_spring_stiffness_mean"],
                                          self.config["object_spring_stiffness_sigma"],
                                          self.config["object_elastic_stiffness_mean"],
                                          self.config["object_elastic_stiffness_sigma"],
                                          self.config["object_pybullet_rgba_color"])

        self.object = None

        # start the simulation
        self.start_sim()

        # calculate camera position
        self.setup_camera()

    def get_observations(self, action: PushAction):
        raise NotImplementedError("Method needs to be implemented.")

    def step_sim_with_force(self, action: PushAction):
        def step():
            p.setJointMotorControl2(self.scene["pusher"], 1, p.POSITION_CONTROL, targetPosition=-1,
                                    force=action.force, maxVelocity=self.config["pusher_lin_vel"])
            p.stepSimulation()

        if self.config["realtime"]:
            t_start = time.time()
            t_end = t_start + self.config["simulation_action_steps"] * self.config["simulation_timestep"]
            while time.time() < t_end:
                step()
                time.sleep(self.config["simulation_timestep"])
        else:
            i = 0
            while i < self.config["simulation_action_steps"]:
                step()
                i += 1

    def start_sim(self):
        if self.config["simulation_use_gui"]:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        if self.config["realtime"]:
            p.setRealTimeSimulation(1)

        self.reset_sim()

    def reset_sim(self):
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setGravity(0, 0, GRAVITY)

        self.scene["plane"] = self.setup_scene()
        self.scene["pusher"] = None  # pusher is respawned on each get_observation()

        try:
            self.object = self.rog.generate_object()
        except ValueError as e:
            log(TextFlag.ERROR, e)

        p.stepSimulation()

    def stop_sim(self):
        p.disconnect()

    def setup_scene(self):
        plane_id = p.createMultiBody(p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[100, 100, 0.1]), 0)
        p.changeDynamics(bodyUniqueId=plane_id, linkIndex=-1, mass=0, restitution=1.0,
                         lateralFriction=1.0, rollingFriction=1.0, spinningFriction=1.0,
                         contactDamping=-1, contactStiffness=-1)
        p.changeVisualShape(objectUniqueId=plane_id, linkIndex=-1, rgbaColor=[0.3, 0.3, 0.3, 1])
        return plane_id

    def setup_pusher(self, object_pos=None, action: PushAction = None):
        # create collision boxes
        base = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
        link = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.02, 0.02])
        pusher = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.05, 0.1])

        # pusher position around the object on a circle
        pos_offset = self.config["object_position"] + np.array(object_pos) \
            if object_pos is not None else self.config["object_position"]

        yaw = action.yaw if action is not None else 0.0
        base_position, base_orientation = pose_on_circle(radius=self.config["pusher_radius"],
                                                         yaw=yaw,
                                                         height=self.config["pusher_height"],
                                                         pos_offset=pos_offset)

        baseMass = 0  # fixed base
        baseCollisionShapeIndex = base
        baseVisualShapeIndex = -1
        linkMasses = [self.config["pusher_link_mass"]] * 3
        linkCollisionShapeIndices = [-1, link, pusher]
        linkVisualShapeIndices = [-1, link, pusher]
        linkPositions = [[0, 0, 0], [-0.1, 0, 0], [-0.1, 0, 0]]
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

        # make prismatic joint more springy
        p.changeDynamics(pusher_id, 3, linearDamping=1.0, angularDamping=1.0, jointDamping=1.0)

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

        return pusher_id, base_position, base_orientation

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

    def get_color_image(self, color=True, raw=False):
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

    def get_depth_image(self):
        img_arr = p.getCameraImage(self.config["projection_w"],
                                   self.config["projection_h"],
                                   self.viewMatrix,
                                   self.projectionMatrix)

        d = img_arr[3].astype(np.float32)
        d = self.config["far_plane"] * self.config["near_plane"] / \
            (self.config["far_plane"] - (self.config["far_plane"] -
                                         self.config["near_plane"]) * d)

        return d[..., np.newaxis]  # add number of channels
