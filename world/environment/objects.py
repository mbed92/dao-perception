import os

import numpy as np
import pybullet as p
import yaml

YAML_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'objects.yaml')


class RandomObjectsGenerator:
    def __init__(self,
                 position_mean=None, position_sigma=None,
                 orientation=None,
                 size_mean=None, size_sigma=None,
                 mass_mean=None, mass_sigma=None,
                 friction_mean=None, friction_sigma=None,
                 restitution_mean=None, restitution_sigma=None,
                 movable=None):
        self.position_mean = [0.0, 0.0, 0.1] if position_mean is None else position_mean
        self.position_sigma = [0.0, 0.0, 0.0] if position_sigma is None else position_sigma
        self.orientation = [0, 0, 0, 1] if orientation is None else orientation
        self.size_mean = 1.0 if size_mean is None else size_mean
        self.size_sigma = 0.2 if size_sigma is None else size_sigma
        self.mass_mean = 1.0 if mass_mean is None else mass_mean
        self.mass_sigma = 0.5 if mass_sigma is None else mass_sigma
        self.friction_mean = 0.2 if friction_mean is None else friction_mean
        self.friction_sigma = 0.1 if friction_sigma is None else friction_sigma
        self.restitution_mean = 1.0 if restitution_mean is None else restitution_mean
        self.restitution_sigma = 0.9 if restitution_sigma is None else restitution_sigma
        self.movable = True if movable is None else movable

        assert len(self.position_mean) == 3
        assert len(self.position_sigma) == 3
        assert len(self.orientation) == 4
        assert type(self.size_mean) is float
        assert type(self.size_sigma) is float
        assert type(self.mass_mean) is float
        assert type(self.mass_sigma) is float
        assert type(self.friction_mean) is float
        assert type(self.friction_sigma) is float
        assert type(self.restitution_mean) is float
        assert type(self.restitution_sigma) is float
        assert type(self.movable) is bool

        stream = open(YAML_CONFIG_PATH, 'r')
        self.objects_list = yaml.safe_load(stream)

        self.MASS_ADJECTIVES = ["light", "medium_mass", "heavy"]
        self.FRICTION_ADJECTIVES = ["slippery", "smooth", "rough"]
        self.RESTITUTION_ADJECTIVES = ["soft", "springy", "hard"]
        self.MOVABLE_ADJECTIVES = ["fixed", "movable"]
        self.HAPTIC_ADJECTIVES = self.MASS_ADJECTIVES + self.FRICTION_ADJECTIVES + \
                                 self.RESTITUTION_ADJECTIVES + self.MOVABLE_ADJECTIVES
        self.mass = None
        self.friction = None
        self.restitution = None

    def get_haptic_adjectives(self):
        adjectives = list()

        if self.mass is not None:
            mass_ranges = [
                [0.0, self.mass_mean - 0.25 * self.mass_sigma],
                [self.mass_mean - 0.25 * self.mass_sigma, self.mass_mean + 0.25 * self.mass_sigma],
                [self.mass_mean + 0.25 * self.mass_sigma, 999.9]
            ]
            adjectives.append(
                [self.MASS_ADJECTIVES[i] for i, r in enumerate(mass_ranges) if r[0] < self.mass < r[1]][0])

        if self.friction is not None:
            friction_ranges = [
                [0.0, self.friction_mean - 0.25 * self.friction_sigma],
                [self.friction_mean - 0.25 * self.friction_sigma, self.friction_mean + 0.25 * self.friction_sigma],
                [self.friction_mean + 0.25 * self.friction_sigma, 999.9]
            ]
            adjectives.append(
                [self.FRICTION_ADJECTIVES[i] for i, r in enumerate(friction_ranges) if r[0] < self.friction < r[1]][0])

        if self.restitution is not None:
            restitution_ranges = [
                [0.0, self.restitution_mean - 0.25 * self.restitution_sigma],
                [self.restitution_mean - 0.25 * self.restitution_sigma,
                 self.restitution_mean + 0.25 * self.restitution_sigma],
                [self.restitution_mean + 0.25 * self.restitution_sigma, 999.9]
            ]
            adjectives.append(
                [self.RESTITUTION_ADJECTIVES[i] for i, r in enumerate(restitution_ranges) if
                 r[0] < self.restitution < r[1]][0])

        if self.movable:
            adjectives.append(self.MOVABLE_ADJECTIVES[1])
        else:
            adjectives.append(self.MOVABLE_ADJECTIVES[0])

        if len(adjectives) == 0:
            print("Create object before checking its haptic properties.")

        return adjectives

    def generate_object(self, flags=0):
        obj_position = [np.random.uniform(m - s, m + s) for m, s in zip(self.position_mean, self.position_sigma)]
        obj_scale = np.random.uniform(self.size_mean - self.size_sigma, self.size_mean + self.size_sigma)
        self.mass = np.random.uniform(self.mass_mean + self.mass_sigma, self.mass_mean - self.mass_sigma)
        self.friction = np.random.uniform(self.friction_mean + self.friction_sigma,
                                          self.friction_mean - self.friction_sigma)
        self.restitution = np.random.uniform(self.restitution_mean + self.restitution_sigma,
                                             self.restitution_mean - self.restitution_sigma)
        self.movable = True if np.random.rand() < 0.8 else False

        # load the object
        obj_to_load = np.random.choice(self.objects_list)
        obj_id = p.loadURDF(obj_to_load, obj_position, self.orientation,
                            useFixedBase=self.movable,
                            flags=flags,
                            globalScaling=obj_scale)

        # do not assign mass adjective if object is fixed
        if self.movable:
            p.changeDynamics(mass=self.mass, restitution=self.restitution, bodyUniqueId=obj_id, linkIndex=-1,
                             lateralFriction=self.friction)
        else:
            self.mass = None
            self.friction = None
            p.changeDynamics(mass=999, restitution=self.restitution, bodyUniqueId=obj_id, linkIndex=-1,
                             lateralFriction=999)

        return obj_id
