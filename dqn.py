"""Defines deep Q-network architectures.

Heavily influenced by DeepMind's seminal paper 'Playing Atari with Deep Reinforcement Learning'
(Mnih et al., 2013) and 'Human-level control through deep reinforcement learning' (Mnih et al.,
2015).
"""

import abc
import math
import numpy as np
import tensorflow as tf


def _convolutional_layer(x, shape, stride, activation_fn):
    if len(shape) != 4:
        raise ValueError('Shape "{}" is invalid. Must have length 4.'.format(shape))

    num_input_params = shape[0] * shape[1] * shape[2]
    num_output_params = shape[0] * shape[1] * shape[3]
    maxval = math.sqrt(6 / (num_input_params + num_output_params))
    W = tf.Variable(tf.random_uniform(shape, -maxval, maxval), name='Weights')
    b = tf.Variable(tf.constant(0.1, tf.float32, [shape[3]]), name='Bias')
    conv = tf.nn.conv2d(x, W, [1, stride, stride, 1], 'VALID')

    return activation_fn(tf.nn.bias_add(conv, b))


def _fully_connected_layer(x, shape, bias_shape, activation_fn):
    if len(shape) != 2:
        raise ValueError('Shape "{}" is invalid. Must have length 2.'.format(shape))

    maxval = 1 / math.sqrt(shape[0] + shape[1])
    W = tf.Variable(tf.random_uniform(shape, -maxval, maxval), name='Weights')
    b = tf.Variable(tf.constant(0.1, tf.float32, bias_shape), name='Bias')

    return activation_fn(tf.matmul(x, W) + b)


class _QNetwork(abc.ABC):
    """Base class for neural networks that learn the Q (action value) function."""

    @abc.abstractmethod
    def __init__(self):
        """Creates a Q-network."""

    def get_action_value(self, state, action):
        """Estimates the value of the specified action for the specified state.

        Args:
            state: State of the environment. Can be batched into multiple states.
            action: A valid action. Can be batched into multiple actions.
        """

        sess = tf.get_default_session()
        return sess.run(self.estimated_action_value, {self.x: state, self.action: action})

    def get_optimal_action_value(self, state):
        """Estimates the optimal action value for the specified state.

        Args:
            state: State of the environment. Can be batched into multiple states.
        """

        sess = tf.get_default_session()
        return sess.run(self.optimal_action_value, {self.x: state})

    def get_optimal_action(self, state):
        """Estimates the optimal action for the specified state.

        Args:
            state: State of the environment. Can be batched into multiple states.
        """

        sess = tf.get_default_session()
        return sess.run(self.optimal_action, {self.x: state})


class DeepQNetwork(_QNetwork):
    """A deep Q-network."""

    def __init__(self, state_shape, num_actions):
        """Creates a deep Q-network.

        Args:
            state_shape: A vector with three values, representing the width, height and depth of
                input states. For example, the shape of 100x80 RGB images is [100, 80, 3].
            num_actions: Number of possible actions.
        """

        width, height, depth = state_shape
        self.x = tf.placeholder(tf.float32, [None, width, height, depth], name='Input_States')

        with tf.name_scope('Convolutional_Layer_1'):
            h_conv1 = _convolutional_layer(self.x, [4, 4, depth, 64], 2, tf.nn.relu)

        with tf.name_scope('Convolutional_Layer_2'):
            h_conv2 = _convolutional_layer(h_conv1, [3, 3, 64, 64], 2, tf.nn.relu)

        with tf.name_scope('Convolutional_Layer_3'):
            h_conv3 = _convolutional_layer(h_conv2, [3, 3, 64, 64], 1, tf.nn.relu)

        # Flatten the output to feed it into fully connected layers.
        num_params = np.prod(h_conv3.get_shape().as_list()[1:])
        h_flat = tf.reshape(h_conv3, [-1, num_params])

        with tf.name_scope('Fully_Connected_Layer_1'):
            h_fc = _fully_connected_layer(h_flat, [num_params, 512], [512], tf.nn.relu)

        with tf.name_scope('Fully_Connected_Layer_2'):
            # Use a single shared bias for each action.
            self.Q = _fully_connected_layer(h_fc, [512, num_actions], [1], tf.identity)

        # Estimate the optimal action and its expected value.
        self.optimal_action = tf.squeeze(tf.argmax(self.Q, 1, name='Optimal_Action'))
        self.optimal_action_value = tf.squeeze(tf.reduce_max(self.Q, 1))

        # Estimate the value of the specified action.
        self.action = tf.placeholder(tf.uint8, name='Action')
        one_hot_action = tf.one_hot(self.action, num_actions)
        self.estimated_action_value = tf.reduce_sum(self.Q * one_hot_action, 1)


class DuelingDeepQNetwork(_QNetwork):
    """A deep Q-network with a dueling architecture."""

    def __init__(self, state_shape, num_actions):
        """Creates a deep Q-network with a dueling architecture.

        Args:
            state_shape: A vector with three values, representing the width, height and depth of
                input states. For example, the shape of 100x80 RGB images is [100, 80, 3].
            num_actions: Number of possible actions.
        """

        width, height, depth = state_shape
        self.x = tf.placeholder(tf.float32, [None, width, height, depth], name='Input_States')

        with tf.name_scope('Convolutional_Layer_1'):
            h_conv1 = _convolutional_layer(self.x, [4, 4, depth, 64], 2, tf.nn.relu)

        with tf.name_scope('Convolutional_Layer_2'):
            h_conv2 = _convolutional_layer(h_conv1, [3, 3, 64, 64], 2, tf.nn.relu)

        with tf.name_scope('Convolutional_Layer_3'):
            h_conv3 = _convolutional_layer(h_conv2, [3, 3, 64, 64], 1, tf.nn.relu)

        # Flatten the output to feed it into fully connected layers.
        num_params = np.prod(h_conv3.get_shape().as_list()[1:])
        h_flat = tf.reshape(h_conv3, [-1, num_params])

        with tf.name_scope('Advantage_Stream'):
            h_advantage_fc = _fully_connected_layer(h_flat, [num_params, 512], [512], tf.nn.relu)

            # Use a single shared bias for each action.
            advantage = _fully_connected_layer(h_advantage_fc, [512, num_actions], [1], tf.identity)

        with tf.name_scope('State_Value_Stream'):
            h_state_value_fc = _fully_connected_layer(h_flat, [num_params, 512], [512], tf.nn.relu)
            state_value = _fully_connected_layer(h_state_value_fc, [512, 1], [1], tf.identity)

        # Connect streams and estimate action values (Q). To improve training stability as suggested
        # by Wang et al., 2015, Q = state value + advantage - mean(advantage).
        self.Q = state_value + advantage - tf.reduce_mean(advantage, 1, keep_dims=True)

        # Estimate the optimal action and its expected value.
        self.optimal_action = tf.squeeze(tf.argmax(self.Q, 1, name='Optimal_Action'))
        self.optimal_action_value = tf.squeeze(tf.reduce_max(self.Q, 1))

        # Estimate the value of the specified action.
        self.action = tf.placeholder(tf.uint8, name='Action')
        one_hot_action = tf.one_hot(self.action, num_actions)
        self.estimated_action_value = tf.reduce_sum(self.Q * one_hot_action, 1)
