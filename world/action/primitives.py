import numpy as np


class PushAction:
    def __init__(self, yaw, force):
        super().__init__()
        self.force = force
        self.yaw = yaw

    def to_numpy(self):
        return np.asarray([self.yaw, self.force])[np.newaxis, ...]

    def __repr__(self):
        return f"Action():\t" \
               f"yaw={self.yaw}\tforce={self.force}"

    @classmethod
    def random_sample(cls, force_x=10):
        yaw = np.random.uniform(-np.pi, np.pi)
        force = np.random.uniform(force_x - 0.5 * force_x, force_x + 0.5 * force_x)
        return cls(yaw, force)
