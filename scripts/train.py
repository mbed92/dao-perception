# import gym, ray
# from ray.rllib.agents import ppo
#
# class MyEnv(gym.Env):
#     def __init__(self, env_config):
#         self.action_space = <gym.Space>
#         self.observation_space = <gym.Space>
#     def reset(self):
#         return <obs>
#     def step(self, action):
#         return <obs>, <reward: float>, <done: bool>, <info: dict>
#
# ray.init()
# trainer = ppo.PPOTrainer(env=MyEnv, config={
#     "env_config": {},  # config to pass to env class
# })
#
# while True:
#     print(trainer.train())