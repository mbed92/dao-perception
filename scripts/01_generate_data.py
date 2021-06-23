import yaml

import world

NUM_EPISODES = 10
ENV_CONFIG = yaml.safe_load(open("../config/env.yaml", 'r'))

if __name__ == "__main__":
    myenv = world.environment.pusher.PusherEnvGenerator(ENV_CONFIG)

    for _ in range(NUM_EPISODES):
        myenv.reset()
        action = world.action.primitives.PushAction.random_sample()
        observations, reward, done, info = myenv.step(action=action)
        print(info)

    myenv.stop_sim()
