import os

import numpy as np
import pybullet as p
import yaml

YAML_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'objects.yaml')


class RandomObjectsGenerator:
    def __init__(self, mean_x=0.0, mean_y=0.0, mean_z=0.0,
                 sigma_x=0.1, sigma_y=0.1, sigma_z=0.1,
                 globalScaling_mean=1.0, globalScaling_signa=0.1,
                 rand_x=True, rand_y=True, rand_z=True,
                 additional_objects_folder=None):

        self.mx = mean_x
        self.sx = sigma_x
        self.my = mean_y
        self.sy = sigma_y
        self.mz = mean_z
        self.sz = sigma_z
        self.rand_x = rand_x
        self.rand_y = rand_y
        self.rand_z = rand_z
        self.globalScaling_mean = globalScaling_mean
        self.globalScaling_sigma = globalScaling_signa

        if additional_objects_folder is not None and os.path.exists(additional_objects_folder):
            p.setAdditionalSearchPath(additional_objects_folder)
        else:
            print(f'Directory {additional_objects_folder} does not exist.')

        stream = open(YAML_CONFIG_PATH, 'r')
        self.available_objects_urdf = yaml.safe_load(stream)

    def __call__(self, flags=0):
        x, y, z = self.mx, self.my, self.mz
        if self.rand_x:
            x = np.random.normal(self.mx, self.sx)
        if self.rand_y:
            y = np.random.normal(self.my, self.sy)
        if self.rand_z:
            z = np.random.normal(self.mz, self.sz)

        object_to_load = np.random.choice(self.available_objects_urdf)
        object_scale = np.random.normal(self.globalScaling_mean, self.globalScaling_sigma)
        object_id = p.loadURDF(object_to_load, [x, y, z], flags=flags, globalScaling=object_scale)
        return object_id
