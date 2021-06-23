import numpy as np


class PushAction:
    def __init__(self, yaw, force):
        self.force = force
        self.yaw = yaw

    def __repr__(self):
        return f"Action():\t" \
               f"start point={self.yaw}\tend point={self.force}\tforce={self.force}\t"

    @classmethod
    def random_sample(cls, yaw_zero=0.0, yaw_delta=0.1, force_zero=5, force_delta=2.0):
        yaw = np.random.uniform(yaw_zero - yaw_delta, yaw_zero + yaw_delta)
        force = np.random.uniform(force_zero - force_delta, force_zero + force_delta)
        return cls(yaw, force)
