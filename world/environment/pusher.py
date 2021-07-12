from ray.rllib.env import EnvContext

from world.action.primitives import PushAction
from world.environment.base import BaseEnv
from world.physics.phys_net import HapticRegressor


class PusherEnvGenerator(BaseEnv):
    def __init__(self, config: EnvContext):
        super().__init__(config)
        self.haptic_regressor = HapticRegressor(batch_size=config["batch_size"],
                                                num_outputs=config["num_outputs"],
                                                action_kernel_size=config["action_kernel_size"],
                                                dropout=config["dropout"],
                                                lstm_units=config["lstm_units"])

    def step(self, action: PushAction = None):
        assert type(action) is PushAction and self.object is not None
        observations, reward, done, info = list(), None, False, {}

        info["haptic"] = self.rog.get_haptic_values()
        observations = self.get_observations(action)

        obs_flat = observations.reshape((1, -1))
        act_flat = action.to_numpy().reshape((1, -1))
        info["observations_numpy"] = (obs_flat, act_flat)

        return observations, reward, done, info

    def reset(self):
        self.reset_sim()
