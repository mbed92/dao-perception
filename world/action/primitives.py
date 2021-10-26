import numpy as np


class Action(object):
    def to_numpy(self):
        raise NotImplementedError("Implement that method.")

    @staticmethod
    def from_numpy(action):
        raise NotImplementedError("Implement that method.")

    @classmethod
    def random_sample(cls, *args):
        raise NotImplementedError("Implement that method.")


class PushAction(Action):
    def __init__(self, yaw, force):
        super().__init__()
        self.force = force
        self.yaw = yaw

    def to_numpy(self):
        return np.asarray([self.yaw,
                           self.force])[np.newaxis, ...]

    @staticmethod
    def from_numpy(action):
        assert len(action.shape) <= 2
        action = action.reshape(-1)
        yaw = action[0]
        force = action[1]
        return PushAction(yaw, force)

    def __repr__(self):
        return f"Action():\t" \
               f"yaw={self.yaw}\tforce={self.force}"

    @classmethod
    def random_sample(cls, force_x=10):
        yaw = np.random.uniform(-np.pi, np.pi)
        force = np.random.uniform(force_x - 0.5 * force_x, force_x + 0.5 * force_x)
        return cls(yaw, force)


class RobotAction(Action):
    def __init__(self, dx, dy, dz, dyaw):
        super().__init__()
        self.delta_x = dx
        self.delta_y = dy
        self.delta_z = dz
        self.delta_yaw = dyaw

    def to_numpy(self):
        return np.asarray([self.delta_x,
                           self.delta_y,
                           self.delta_z,
                           self.delta_yaw])[np.newaxis, ...]

    @staticmethod
    def from_numpy(action):
        assert len(action.shape) <= 2
        action = action.reshape(-1)
        yaw = action[0]
        force = action[1]
        return PushAction(yaw, force)

    def __repr__(self):
        return f"Action():\tdelta x={self.delta_x}, delta x={self.delta_y}, delta z={self.delta_x}, " \
               f"delta yaw={self.delta_yaw}"

    @classmethod
    def random_sample(cls):
        increment = 0.01
        increment_range = [-increment, increment]
        dx = np.random.uniform(*increment_range)
        dy = np.random.uniform(*increment_range)
        dz = np.random.uniform(*increment_range)
        dyaw = np.random.uniform(-np.pi / 16, np.pi / 16)
        return cls(dx, dy, dz, dyaw)
