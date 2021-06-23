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
                 friction_mean=None, friction_sigma=None):
        self.position_mean = [0.0, 0.0, 0.1] if position_mean is None else position_mean
        self.position_sigma = [0.0, 0.0, 0.0] if position_sigma is None else position_sigma
        self.orientation = [0, 0, 0, 1] if orientation is None else orientation
        self.size_mean = 1.0 if size_mean is None else size_mean
        self.size_sigma = 0.2 if size_sigma is None else size_sigma
        self.mass_mean = 1.0 if mass_mean is None else mass_mean
        self.mass_sigma = 0.5 if mass_sigma is None else mass_sigma
        self.friction_mean = 0.2 if friction_mean is None else friction_mean
        self.friction_sigma = 0.1 if friction_sigma is None else friction_sigma

        assert len(self.position_mean) == 3
        assert len(self.position_sigma) == 3
        assert len(self.orientation) == 4
        assert type(self.size_mean) is float
        assert type(self.size_sigma) is float
        assert type(self.mass_mean) is float
        assert type(self.mass_sigma) is float
        assert type(self.friction_mean) is float
        assert type(self.friction_sigma) is float

        stream = open(YAML_CONFIG_PATH, 'r')
        self.objects_list = yaml.safe_load(stream)

        self.MASS_ADJECTIVES = ["light", "medium_mass", "heavy"]
        self.FRICTION_ADJECTIVES = ["slippery", "medium_friction", "rough"]
        self.mass = None
        self.friction = None

    def get_haptic_adjectives(self):
        adjectives = list()
        if None not in [self.mass, self.friction]:
            mass_ranges = [
                [0.0, self.mass_mean - 0.25 * self.mass_sigma],
                [self.mass_mean - 0.25 * self.mass_sigma, self.mass_mean + 0.25 * self.mass_sigma],
                [self.mass_mean + 0.25 * self.mass_sigma, 999.9]
            ]
            friction_ranges = [
                [0.0, self.friction_mean - 0.25 * self.friction_sigma],
                [self.friction_mean - 0.25 * self.friction_sigma, self.friction_mean + 0.25 * self.friction_sigma],
                [self.friction_mean + 0.25 * self.friction_sigma, 999.9]
            ]

            # gather object info
            adjectives.append(
                [self.MASS_ADJECTIVES[i] for i, r in enumerate(mass_ranges) if r[0] < self.mass < r[1]][0])
            adjectives.append(
                [self.FRICTION_ADJECTIVES[i] for i, r in enumerate(friction_ranges) if r[0] < self.friction < r[1]][0])
        else:
            print("Create object before checking its haptic properties.")

        return adjectives

    def generate_object(self, flags=0):
        obj_to_load = np.random.choice(self.objects_list)
        obj_position = [np.random.uniform(m - s, m + s) for m, s in zip(self.position_mean, self.position_sigma)]
        obj_scale = np.random.uniform(self.size_mean - self.size_sigma, self.size_mean + self.size_sigma)
        obj_id = p.loadURDF(obj_to_load, obj_position, self.orientation, flags=flags, globalScaling=obj_scale)
        self.mass = np.random.uniform(self.mass_mean + self.mass_sigma, self.mass_mean - self.mass_sigma)
        self.friction = np.random.uniform(self.friction_mean + self.friction_sigma,
                                          self.friction_mean - self.friction_sigma)
        p.changeDynamics(mass=self.mass, bodyUniqueId=obj_id, linkIndex=-1, lateralFriction=self.friction)
        return obj_id
