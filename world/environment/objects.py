import os

import numpy as np
import pybullet as p

YAML_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'objects.yaml')


def get_normalized_haptic_value(value, minimum, maximum):
    return (value - minimum) / (maximum - minimum)


class RandomObjectsGenerator:
    def __init__(self,
                 position=None, orientation=None,
                 size_mean=None, size_sigma=None,
                 mass_mean=None, mass_sigma=None,
                 friction_mean=None, friction_sigma=None,
                 restitution_mean=None, restitution_sigma=None,
                 spring_stiffness_mean=None, spring_stiffness_sigma=None,
                 elastic_stiffness_mean=None, elastic_stiffness_sigma=None):

        self.position = [0.0, 0.0, 0.1] if position is None else position
        self.orientation = [0, 0, 0, 1] if orientation is None else orientation
        self.size_mean = 1.0 if size_mean is None else size_mean
        self.size_sigma = 0.2 if size_sigma is None else size_sigma
        self.mass_mean = 1.0 if mass_mean is None else mass_mean
        self.mass_sigma = 0.2 if mass_sigma is None else mass_sigma
        self.friction_mean = 0.5 if friction_mean is None else friction_mean
        self.friction_sigma = 0.4 if friction_sigma is None else friction_sigma
        self.restitution_mean = 1.0 if restitution_mean is None else restitution_mean
        self.restitution_sigma = 0.9 if restitution_sigma is None else restitution_sigma
        self.spring_stiffness_mean = 100.0 if spring_stiffness_mean is None else spring_stiffness_mean
        self.spring_stiffness_sigma = 90.0 if spring_stiffness_sigma is None else spring_stiffness_sigma
        self.elastic_stiffness_mean = 100.0 if elastic_stiffness_mean is None else elastic_stiffness_mean
        self.elastic_stiffness_sigma = 90.0 if elastic_stiffness_sigma is None else elastic_stiffness_sigma

        assert len(self.position) == 3
        assert len(self.orientation) == 4
        assert type(self.size_mean) is float
        assert type(self.size_sigma) is float
        assert type(self.mass_mean) is float
        assert type(self.mass_sigma) is float
        assert type(self.friction_mean) is float
        assert type(self.friction_sigma) is float
        assert type(self.restitution_mean) is float
        assert type(self.restitution_sigma) is float
        assert type(self.spring_stiffness_mean) is float
        assert type(self.spring_stiffness_sigma) is float
        assert type(self.elastic_stiffness_mean) is float
        assert type(self.elastic_stiffness_sigma) is float
        self.object_types = ["cube.obj"]

        # physical properies
        self.obj_size = None
        self.mass = None
        self.friction = None
        self.restitution = None
        self.spring_stiffness = None
        self.elastic_stiffness = None

    def get_haptic_values(self):
        haptic = dict()

        if self.mass is not None and self.friction is not None:
            norm_mass = get_normalized_haptic_value(value=self.mass,
                                                    minimum=self.mass_mean - self.mass_sigma,
                                                    maximum=self.mass_mean + self.mass_sigma)
            norm_fri = get_normalized_haptic_value(value=self.friction,
                                                   minimum=self.friction_mean - self.friction_sigma,
                                                   maximum=self.friction_mean + self.friction_sigma)
            norm_mass_fri = max(norm_mass, norm_fri)
            haptic["mass_friction"] = norm_mass_fri

        if self.restitution is not None:
            norm_resti = get_normalized_haptic_value(value=self.restitution,
                                                     minimum=self.restitution_mean - self.restitution_sigma,
                                                     maximum=self.restitution_mean + self.restitution_sigma)
            haptic["restitution"] = norm_resti

        if self.spring_stiffness is not None:
            norm_stiff = get_normalized_haptic_value(value=self.spring_stiffness,
                                                     minimum=self.spring_stiffness_mean - self.spring_stiffness_sigma,
                                                     maximum=self.spring_stiffness_mean + self.spring_stiffness_sigma)
            haptic["stiffness"] = norm_stiff

        if self.elastic_stiffness is not None:
            norm_elastic = get_normalized_haptic_value(value=self.elastic_stiffness,
                                                       minimum=self.elastic_stiffness_mean - self.elastic_stiffness_sigma,
                                                       maximum=self.elastic_stiffness_mean + self.elastic_stiffness_sigma)
            haptic["elasticity"] = norm_elastic

        if len(haptic) == 0:
            print("Create object before checking its haptic properties.")

        return haptic

    def randomize_properties(self):
        self.obj_size = np.random.uniform(self.size_mean - self.size_sigma, self.size_mean + self.size_sigma)
        self.mass = np.random.uniform(self.mass_mean + self.mass_sigma, self.mass_mean - self.mass_sigma)
        self.friction = np.random.uniform(self.friction_mean + self.friction_sigma,
                                          self.friction_mean - self.friction_sigma)

        self.restitution = np.random.uniform(self.restitution_mean + self.restitution_sigma,
                                             self.restitution_mean - self.restitution_sigma)

        self.spring_stiffness = np.random.uniform(self.spring_stiffness_mean + self.spring_stiffness_sigma,
                                                  self.spring_stiffness_mean - self.spring_stiffness_sigma)

        self.elastic_stiffness = np.random.uniform(self.elastic_stiffness_mean + self.elastic_stiffness_sigma,
                                                   self.elastic_stiffness_mean - self.elastic_stiffness_sigma)

    def generate_object(self, rand_properies=True):
        if rand_properies:
            self.randomize_properties()

        obj_to_load = np.random.choice(self.object_types)
        obj_id = p.loadSoftBody(obj_to_load, basePosition=self.position, scale=self.obj_size, mass=self.mass,
                                useNeoHookean=0, useBendingSprings=1, useMassSpring=1, useSelfCollision=0,
                                useFaceContact=1, springDampingAllDirections=1, collisionMargin=1e-4,
                                springElasticStiffness=self.elastic_stiffness,
                                springDampingStiffness=self.spring_stiffness,
                                frictionCoeff=self.friction)

        # do not assign mass adjective if object is fixed
        p.changeDynamics(bodyUniqueId=obj_id, linkIndex=-1,
                         mass=self.mass, restitution=self.restitution,
                         spinningFriction=self.friction,
                         rollingFriction=self.friction)
        return obj_id
