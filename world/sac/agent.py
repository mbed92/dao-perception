import tensorflow as tf
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent, tanh_normal_projection_network
from tf_agents.networks import actor_distribution_network
from tf_agents.train.utils import spec_utils, strategy_utils, train_utils


class SAC:
    def __init__(self, collect_env, use_gpu=True):
        self.critic_learning_rate = 3e-4
        self.actor_learning_rate = 3e-4
        self.alpha_learning_rate = 3e-4
        self.target_update_tau = 0.005
        self.target_update_period = 1
        self.gamma = 0.99
        self.reward_scale_factor = 1.0
        self.actor_fc_layer_params = (256, 256)
        self.critic_joint_fc_layer_params = (256, 256)
        self.strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)

        self.observation_spec, self.action_spec, self.time_step_spec = \
            (spec_utils.get_tensor_specs(collect_env))

        with self.strategy.scope():
            self.critic_net = critic_network.CriticNetwork(
                (self.observation_spec, self.action_spec),
                observation_fc_layer_params=None,
                action_fc_layer_params=None,
                joint_fc_layer_params=self.critic_joint_fc_layer_params,
                kernel_initializer='glorot_uniform',
                last_kernel_initializer='glorot_uniform')

        with self.strategy.scope():
            self.actor_net = actor_distribution_network.ActorDistributionNetwork(
                self.observation_spec, self.action_spec,
                fc_layer_params=self.actor_fc_layer_params,
                continuous_projection_net=(
                    tanh_normal_projection_network.TanhNormalProjectionNetwork))

    def create_agent(self):
        with self.strategy.scope():
            self.train_step = train_utils.create_train_step()

            tf_agent = sac_agent.SacAgent(
                self.time_step_spec,
                self.action_spec,
                actor_network=self.actor_net,
                critic_network=self.critic_net,
                actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=self.actor_learning_rate),
                critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=self.critic_learning_rate),
                alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=self.alpha_learning_rate),
                target_update_tau=self.target_update_tau,
                target_update_period=self.target_update_period,
                td_errors_loss_fn=tf.math.squared_difference,
                gamma=self.gamma,
                reward_scale_factor=self.reward_scale_factor,
                train_step_counter=self.train_step)

            tf_agent.initialize()

        return tf_agent
