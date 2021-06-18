import numpy as np

GRAVITY_ACC = -9.80991

class PushAction:
    def __init__(self,
                 start_x=0.0, start_y=0.0, start_z=0.2,
                 end_x=0.1, end_y=0.1, end_z=0.1, push_force=10,
                 object_mass=1, plane_friction=0.1):
        self.start_point = np.asarray((start_x, start_y, start_z))
        self.end_point = np.asarray((end_x, end_y, end_z))
        self.distance = np.linalg.norm(self.end_point - self.start_point)
        self.push_vector = push_force * (self.end_point - self.start_point) / self.distance
        self.object_mass = object_mass
        self.plane_friction = plane_friction
        self.acceleration = self.push_vector / self.object_mass - GRAVITY_ACC * self.plane_friction
        self.action_time = max(np.sqrt((2 * np.abs(self.distance)) / np.abs(self.acceleration)))

    def __repr__(self):
        return f"Action(): start point={self.start_point}\tend point={self.end_point}\t" \
               f"push_vector={self.push_vector}N\taction_time={self.action_time}s\t" \
               f"object_mass={self.object_mass}kg\tplane_friction={self.plane_friction}\t" \
               f"acceleration={self.acceleration}\taction_time={self.action_time}\t" \
               f"distance={self.distance}"

    @classmethod
    def random_sample(cls, x=0.0, y=0.0, z=0.2, delta_x=0.1, delta_y=0.1, delta_z=0.1):
        target_x = np.random.uniform(x - delta_x, x + delta_x)
        target_y = np.random.uniform(y - delta_y, y + delta_y)
        target_z = np.random.uniform(z - delta_z, z + delta_z)
        return cls(end_x=target_x, end_y=target_y, end_z=target_z)
