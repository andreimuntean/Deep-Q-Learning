"""Defines an agent that learns to play Atari games using deep Q-learning.

Heavily influenced by DeepMind's seminal paper 'Playing Atari with Deep Reinforcement Learning'
(Mnih et al., 2013) and 'Human-level control through deep reinforcement learning' (Mnih et al.,
2015).
"""

import dqn
import numpy as np
import random
import tensorflow as tf


class TestOnlyAgent():
    def __init__(self, env):
        """An agent that maximizes its score using deep Q-learning.

        Args:
            env: An AtariWrapper object (see 'environment.py') that wraps over an OpenAI Gym
                environment.
        """

        self.env = env
        self.dqn = dqn.DeepQNetwork(env.state_shape, env.num_actions)

    def get_action(self, state):
        """Estimates the optimal action for the specified state."""

        action_i = self.dqn.get_optimal_action(state)
        return self.env.action_space[action_i]


class Agent():
    def __init__(self,
                 env,
                 start_epsilon,
                 end_epsilon,
                 anneal_duration,
                 train_interval,
                 target_network_reset_interval,
                 batch_size,
                 learning_rate,
                 max_gradient_norm,
                 discount):
        """An agent that learns to play Atari games using deep Q-learning.

        Args:
            env: An AtariWrapper object (see 'environment.py') that wraps over an OpenAI Gym Atari
                environment.
            start_epsilon: Initial value for epsilon (exploration chance) used when training.
            end_epsilon: Final value for epsilon (exploration chance) used when training.
            anneal_duration: Number of time steps needed to decrease epsilon from start_epsilon to
                end_epsilon when training.
            train_interval: Number of experiences to accumulate before another round of training
                starts.
            target_network_reset_interval: Rate at which target Q-network values reset to actual
                Q-network values. Using a delayed target Q-network improves training stability.
            batch_size: Number of experiences sampled and trained on at once.
            learning_rate: The speed with which the network learns from new examples.
            max_gradient_norm: Maximum value allowed for the L2-norms of gradients. Gradients with
                norms that would otherwise surpass this value are scaled down.
            discount: Discount factor for future rewards.
        """

        self.env = env
        self.dqn = dqn.DeepQNetwork(env.state_shape, env.num_actions)
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.anneal_duration = anneal_duration
        self.train_interval = train_interval
        self.target_network_reset_interval = target_network_reset_interval
        self.batch_size = batch_size
        self.discount = discount
        self.time_step = 0
        self.episodes_played = 0
        self.epsilon = self._get_epsilon()

        # Create target Q-network.
        dqn_params = tf.trainable_variables()
        self.target_dqn = dqn.DeepQNetwork(env.state_shape, env.num_actions)
        target_dqn_params = tf.trainable_variables()[len(dqn_params):]

        # Reset target Q-network values to the actual Q-network values.
        self.reset_target_dqn = [old.assign(new) for old, new in zip(target_dqn_params, dqn_params)]

        # Define the optimization scheme for the deep Q-network.
        self.reward = tf.placeholder(tf.float32, [None], name='Observed_Reward')
        self.ongoing = tf.placeholder(tf.bool, [None], name='State_Is_Nonterminal')

        # Determine the true action values using double Q-learning (Hasselt et al., 2015): estimate
        # optimal actions using the Q-network, but estimate their values using the (delayed) target
        # Q-network. This reduces the likelihood that Q is overestimated.
        next_optimal_action_value = tf.stop_gradient(self.target_dqn.estimated_action_value)
        observed_action_value = (
            self.reward + tf.cast(self.ongoing, tf.float32) * discount * next_optimal_action_value)

        # Compute the loss function and regularize it by clipping the norm of its gradients.
        loss = tf.nn.l2_loss(self.dqn.estimated_action_value - observed_action_value)
        gradients = tf.gradients(loss, dqn_params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

        # Perform gradient descent.
        grads_and_vars = list(zip(clipped_gradients, dqn_params))
        self.global_step = tf.Variable(tf.constant(0, tf.int64), False, name='Global_Step')
        self.train_step = tf.train.AdamOptimizer(learning_rate).apply_gradients(
            grads_and_vars, self.global_step)

    def train(self):
        """Performs a single learning step."""

        sess = tf.get_default_session()

        if self.time_step == 0:
            # Initialize target Q-network.
            sess.run(self.reset_target_dqn)

        self.epsilon = self._get_epsilon()
        self.time_step += 1

        # Occasionally try a random action (explore).
        if random.random() < self.epsilon:
            action = self.env.sample_action()
        else:
            action = self.get_action(self.env.get_state())

        self.env.step(action)

        # Occasionally train.
        if self.time_step % self.train_interval == 0:
            # Sample experiences.
            states, actions, rewards, next_states, ongoing = self.env.sample_experiences(self.batch_size)
            actions_i = np.stack([self.env.action_space.index(a) for a in actions], axis=0)
            next_optimal_actions = self.dqn.get_optimal_action(next_states)

            # Estimate action values, measure errors and update weights.
            sess.run(self.train_step, {self.dqn.x: states,
                                       self.dqn.action: actions_i,
                                       self.reward: rewards,
                                       self.target_dqn.x: next_states,
                                       self.target_dqn.action: next_optimal_actions,
                                       self.ongoing: ongoing})

        # Occasionally reset target Q-network values to actual Q-network values.
        if self.time_step % self.target_network_reset_interval == 0:
            sess.run(self.reset_target_dqn)

    def get_action(self, state):
        """Estimates the optimal action for the specified state."""

        action_i = self.dqn.get_optimal_action(state)
        return self.env.action_space[action_i]

    def _get_epsilon(self):
        """Gets the epsilon value (exploration chance) for the current time step."""

        # Epsilon anneals linearly from start_epsilon to end_epsilon.
        if self.anneal_duration <= 0:
            return self.end_epsilon

        epsilon = (self.start_epsilon - self.time_step
                   * (self.start_epsilon - self.end_epsilon) / self.anneal_duration)

        return max(epsilon, self.end_epsilon)
