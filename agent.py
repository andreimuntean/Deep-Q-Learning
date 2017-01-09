"""Defines an agent that learns to play Atari games using deep Q-learning.

Heavily influenced by DeepMind's seminal paper 'Playing Atari with Deep Reinforcement Learning'
(Mnih et al., 2013) and 'Human-level control through deep reinforcement learning' (Mnih et al.,
2015).
"""

import dqn
import math
import numpy as np
import random
import tensorflow as tf


class Agent():
    def __init__(self,
                 sess,
                 env,
                 start_epsilon,
                 end_epsilon,
                 anneal_duration,
                 train_interval,
                 batch_size,
                 learning_rate,
                 dropout_prob,
                 discount,
                 target_network_update_factor):
        """An agent that learns to play Atari games using deep Q-learning.

        Args:
            sess: The associated TensorFlow session.
            env: An AtariWrapper object (see 'environment.py') that wraps over an OpenAI Gym Atari
                environment.
            start_epsilon: Initial value for epsilon (exploration chance) used when training.
            end_epsilon: Final value for epsilon (exploration chance) used when training.
            anneal_duration: Number of time steps needed to decrease epsilon from start_epsilon to
                end_epsilon when training.
            train_interval: Number of experiences to accumulate before another round of training
                starts.
            batch_size: Number of experiences sampled and trained on at once.
            learning_rate: The speed with which the network learns from new examples.
            dropout_prob: Likelihood of neurons from fully connected layers becoming inactive.
            discount: Discount factor for future rewards.
            target_network_update_factor: Rate at which target Q-network values shift toward actual
                Q-network values. This delayed target network improves training stability.
        """

        self.sess = sess
        self.env = env
        self.dqn = dqn.DeepQNetwork(sess, len(env.action_space), env.state_space)
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.anneal_duration = anneal_duration
        self.train_interval = train_interval
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        self.discount = discount
        self.time_step = 0
        self.episodes_played = 0
        self.epsilon = self._get_epsilon()

        # Create target Q-network.
        dqn_params = tf.trainable_variables()
        self.target_dqn = dqn.DeepQNetwork(sess, len(env.action_space), env.state_space)
        target_dqn_params = tf.trainable_variables()[len(dqn_params):]

        # Reset target Q-network values to the actual Q-network values.
        self.reset_target_dqn = [old.assign(new) for old, new in zip(target_dqn_params, dqn_params)]

        # Shift target Q-network values toward actual Q-network values.
        tau = target_network_update_factor
        self.update_target_dqn = [old.assign(tau * new + (1 - tau) * old)
                                  for old, new in zip(target_dqn_params, dqn_params)]

    def train(self):
        """Performs a single learning step."""

        if self.time_step == 0:
            # Initialize target Q-network.
            self.sess.run(self.reset_target_dqn)

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

            # Determine the true action values.
            #
            #                    { r, if next state is terminal
            # Q(state, action) = {
            #                    { r + discount * max(Q(next state, <any action>)), otherwise
            Q_ = rewards + ongoing * self.discount * self.target_dqn.eval_optimal_action_value(next_states)

            # Estimate action values, measure errors and update weights.
            self.dqn.train(states, actions_i, Q_, self.learning_rate, self.dropout_prob)

            # Shift target Q-network values toward actual Q-network values.
            self.sess.run(self.update_target_dqn)

    def get_action(self, state):
        """Estimates the optimal action for the specified state."""

        # Turn the state into a batch of size 1.
        state = np.expand_dims(state, axis=0)

        # Estimate the optimal action index.
        action_i = self.dqn.eval_optimal_action(state)[0]
        action = self.env.action_space[action_i]

        return action

    def _get_epsilon(self):
        """Gets the epsilon value (exploration chance) for the current time step."""

        # Epsilon anneals linearly from start_epsilon to end_epsilon.
        if self.anneal_duration <= 0:
            return self.end_epsilon

        epsilon = (self.start_epsilon - self.time_step
                   * (self.start_epsilon - self.end_epsilon) / self.anneal_duration)

        return max(epsilon, self.end_epsilon)
