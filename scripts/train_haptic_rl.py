import os
import tempfile

import reverb
import yaml
from tf_agents.metrics import py_metrics
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers

import world

tempdir = os.path.join(tempfile.gettempdir(), "rl")

env_name = "MinitaurBulletEnv-v0"  # @param {type:"string"}

# Use "num_iterations = 1e6" for better results (2 hrs)
# 1e5 is just so this doesn't take too long (1 hr)
num_iterations = 100000  # @param {type:"integer"}
initial_collect_steps = 10  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_capacity = 10  # @param {type:"integer"}
batch_size = 16  # @param {type:"integer"}
critic_learning_rate = 3e-4  # @param {type:"number"}
actor_learning_rate = 3e-4  # @param {type:"number"}
alpha_learning_rate = 3e-4  # @param {type:"number"}
target_update_tau = 0.005  # @param {type:"number"}
target_update_period = 1  # @param {type:"number"}
gamma = 0.99  # @param {type:"number"}
reward_scale_factor = 1.0  # @param {type:"number"}
actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)
log_interval = 100  # @param {type:"integer"}
num_eval_episodes = 5  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}
policy_save_interval = 100  # @param {type:"integer"}
visualize_interval = 10

## ENVIRONMENT
# env = suite_pybullet.load(env_name)
# collect_env = suite_pybullet.load(env_name)
# eval_env = suite_pybullet.load(env_name)
# env.reset()

ENV_CONFIG = yaml.safe_load(open("../config/train_haptic_rl.yaml", 'r'))
env = world.environment.pusher.RLPusherEnvGenerator(ENV_CONFIG)
collect_env = world.environment.pusher.RLPusherEnvGenerator(ENV_CONFIG)
eval_env = world.environment.pusher.RLPusherEnvGenerator(ENV_CONFIG)
env.reset()

print('Observation Spec:')
print(env.time_step_spec().observation)
print('Action Spec:')
print(env.action_spec())

## AGENT
sac = world.sac.agent.SAC(collect_env)
tf_agent = sac.create_agent()

### REPLAY BUFFER
rate_limiter = reverb.rate_limiters.SampleToInsertRatio(samples_per_insert=3.0, min_size_to_sample=3, error_buffer=3.0)
table_name = 'uniform_table'
table = reverb.Table(
    table_name,
    max_size=replay_buffer_capacity,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1))

reverb_server = reverb.Server([table])
reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    sequence_length=2,
    table_name=table_name,
    local_server=reverb_server)
dataset = reverb_replay.as_dataset(sample_batch_size=batch_size, num_steps=2).prefetch(5)
experience_dataset_fn = lambda: dataset

### POLICIES
eval_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_agent.policy, use_tf_function=True)
collect_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_agent.collect_policy, use_tf_function=True)
random_policy = random_py_policy.RandomPyPolicy(collect_env.time_step_spec(), collect_env.action_spec())

### ACTORS
rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    reverb_replay.py_client,
    table_name,
    sequence_length=2,
    stride_length=1)

initial_collect_actor = actor.Actor(
    collect_env,
    random_policy,
    sac.train_step,
    steps_per_run=initial_collect_steps,
    observers=[rb_observer])
initial_collect_actor.run()

env_step_metric = py_metrics.EnvironmentSteps()
collect_actor = actor.Actor(
    collect_env,
    collect_policy,
    sac.train_step,
    steps_per_run=1,
    metrics=actor.collect_metrics(10),
    summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
    observers=[rb_observer, env_step_metric])

eval_actor = actor.Actor(
    eval_env,
    eval_policy,
    sac.train_step,
    episodes_per_run=num_eval_episodes,
    metrics=actor.eval_metrics(num_eval_episodes),
    summary_dir=os.path.join(tempdir, 'eval'),
)

### LEARNERS
saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

# Triggers to save the agent's policy checkpoints.
learning_triggers = [
    triggers.PolicySavedModelTrigger(
        saved_model_dir,
        tf_agent,
        sac.train_step,
        interval=policy_save_interval),
    triggers.StepPerSecondLogTrigger(sac.train_step, interval=100),
]

agent_learner = learner.Learner(
    tempdir,
    sac.train_step,
    tf_agent,
    experience_dataset_fn,
    triggers=learning_triggers)


### METRICS
def get_eval_metrics():
    eval_actor.run()
    results = {}
    for metric in eval_actor.metrics:
        results[metric.name] = metric.result()
    return results


metrics = get_eval_metrics()


def log_eval_metrics(step, metrics):
    eval_results = (', ').join(
        '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
    print('step = {0}: {1}'.format(step, eval_results))


log_eval_metrics(0, metrics)

### TRAINING
# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = get_eval_metrics()["AverageReturn"]
returns = [avg_return]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plt.ion()

for _ in range(num_iterations):

    # Training.
    collect_actor.run()
    loss_info = agent_learner.run(iterations=1)

    # Evaluating
    if eval_interval and agent_learner.train_step_numpy % eval_interval == 0:
        metrics = get_eval_metrics()
        log_eval_metrics(agent_learner.train_step_numpy, metrics)
        returns.append(metrics["AverageReturn"])

    if log_interval and agent_learner.train_step_numpy % log_interval == 0:
        print('step = {0}: loss = {1}'.format(agent_learner.train_step_numpy, loss_info.loss.numpy()))

    if visualize_interval and agent_learner.train_step_numpy % visualize_interval == 0:
        time_step = eval_env.reset()
        while not time_step.is_last():
            action_step = eval_actor.policy.action(time_step)
            time_step = eval_env.step(action_step.action)
            img = eval_env.get_camera_image()
            plt.imshow(img)
            plt.show(block=False)
            plt.pause(0.00001)

rb_observer.close()
reverb_server.stop()
