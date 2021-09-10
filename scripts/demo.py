import yaml

import world
from utils.text import TextFlag, log

NUM_EPISODES = 10
NUM_ACTIONS = 10
ENV_CONFIG = yaml.safe_load(open("../config/demo.yaml", 'r'))

if __name__ == "__main__":
    myenv = world.environment.pusher_demo.PusherDemo(ENV_CONFIG)
    myenv.rog.object_types = ['cube.obj']
    myenv.reset()

    for _ in range(NUM_EPISODES):
        for _ in range(NUM_ACTIONS):
            action = world.action.primitives.PushAction.random_sample()
            observations, reward, done, info = myenv.step(action=action)
            log(TextFlag.INFO, info["observations_numpy"])

        myenv.reset()

    myenv.stop_sim()
