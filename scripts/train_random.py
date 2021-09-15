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
env = world.environment.pusher_haptic_encoder_decoder.PushNetEncoderDecoder(ENV_CONFIG)
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

for i in range(1000):
    env.step(action_step.action)

    if i % 20 == 0:
        env.reset()