from ray.rllib.env import EnvContext

from world.action.primitives import PushAction
from world.environment.base import BaseEnv


class PusherEnvGenerator(BaseEnv):
    def __init__(self, config: EnvContext):
        super().__init__(config)

    def step(self, action: PushAction = None):
        assert type(action) is PushAction and self.object is not None
        observations, reward, done, info = list(), None, False, {}
        observations = self.get_observations(action)
        info["haptic"] = self.rog.get_haptic_values()
        return observations, reward, done, info

    def reset(self):
        self.reset_sim()
