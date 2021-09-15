import os
import tempfile

import matplotlib.pyplot as plt
import reverb
import yaml
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.networks import network

from tf_agents.policies import py_policy
from tf_agents.policies import random_py_policy
from tf_agents.policies import scripted_py_policy

from tf_agents.policies import tf_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import actor_policy
from tf_agents.policies import q_policy
from tf_agents.policies import greedy_policy

from tf_agents.trajectories import time_step as ts

import world
from utils.text import TextFlag, log

tempdir = os.path.join(tempfile.gettempdir(), "rl")
plt.ion()

## PARAMS
num_iterations = 100000
initial_collect_steps = 100
collect_steps_per_iteration = 1
replay_buffer_capacity = 100
batch_size = 256
critic_learning_rate = 3e-4
actor_learning_rate = 3e-4
alpha_learning_rate = 3e-4
target_update_tau = 0.005
target_update_period = 1
gamma = 0.99
reward_scale_factor = 1.0
actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)
log_interval = 100
num_eval_episodes = 20
eval_interval = 1000
policy_save_interval = 5000
visualization_on = True
visualize_interval = 10


## ENVIRONMENT
ENV_CONFIG = yaml.safe_load(open("../config/train_haptic_rl.yaml", 'r'))
env = world.environment.pus(ENV_CONFIG)
env.reset()

log(TextFlag.INFO, 'Observation Spec:')
log(TextFlag.INFO, env.time_step_spec().observation)
log(TextFlag.INFO, 'Action Spec:')
log(TextFlag.INFO, env.action_spec())


time_step_spec = ts.time_step_spec(env.observation_spec())
my_random_tf_policy = random_tf_policy.RandomTFPolicy(action_spec=env.action_spec(), time_step_spec=time_step_spec)
observation = tf.ones(time_step_spec.observation.shape)
time_step = ts.restart(observation)
action_step = my_random_tf_policy.action(time_step)

print('Action:')
print(action_step.action)


# ## METRICS
# def get_eval_metrics():
#     eval_actor.run()
#     results = {}
#     for metric in eval_actor.metrics:
#         results[metric.name] = metric.result()
#     return results
#
#
# def log_eval_metrics(step, metrics):
#     eval_results = (', ').join(
#         '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
#     log(TextFlag.INFO, 'step = {0}: {1}'.format(step, eval_results))
#
#
# metrics = get_eval_metrics()
# log_eval_metrics(0, metrics)
#
# ### TRAINING
# # Reset the train step
# log(TextFlag.WARNING, 'Start Training')
# tf_agent.train_step_counter.assign(0)
#
# # Evaluate the agent's policy once before training.
# avg_return = get_eval_metrics()["AverageReturn"]
# returns = [avg_return]
#
# for i in range(num_iterations):
#
#     # Training.
#     collect_actor.run()
#     loss_info = agent_learner.run(iterations=1)
#
#     # Evaluating
#     if eval_interval and agent_learner.train_step_numpy % eval_interval == 0:
#         metrics = get_eval_metrics()
#         log_eval_metrics(agent_learner.train_step_numpy, metrics)
#         returns.append(metrics["AverageReturn"])
#
#     if log_interval and agent_learner.train_step_numpy % log_interval == 0:
#         log(TextFlag.INFO, 'step = {0}: loss = {1}'.format(agent_learner.train_step_numpy, loss_info.loss.numpy()))
#
#     if visualization_on and visualize_interval and agent_learner.train_step_numpy % visualize_interval == 0:
#         log(TextFlag.WARNING, "Visualization on")
#         time_step = eval_env.reset()
#         fig, ax = plt.subplots()
#         while not time_step.is_last():
#             action_step = eval_actor.policy.action(time_step)
#             time_step = eval_env.step(action_step.action)
#             img = eval_env.get_color_image()
#             plt.imshow(img)
#             plt.show(block=False)
#             plt.pause(0.00001)
#         plt.close(fig)
#
# rb_observer.close()
# reverb_server.stop()
