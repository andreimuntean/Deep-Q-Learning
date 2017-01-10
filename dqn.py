"""Defines a deep Q network architecture.

Heavily influenced by DeepMind's seminal paper 'Playing Atari with Deep Reinforcement Learning'
(Mnih et al., 2013) and 'Human-level control through deep reinforcement learning' (Mnih et al.,
2015).
"""

import math
import tensorflow as tf


def _create_fc_weights(shape):
    # Use Xavier initialization.
    minval = -math.sqrt(6.0 / (shape[0] + shape[1]))
    maxval = math.sqrt(6.0 / (shape[0] + shape[1]))
    value = tf.random_uniform(shape, minval, maxval)
    return tf.Variable(value, name='Weights')


def _create_conv2d_weights(shape):
    # Use Xavier initialization.
    minval = -math.sqrt(6.0 / (shape[0] * shape[1] * shape[2] + shape[3]))
    maxval = math.sqrt(6.0 / (shape[0] * shape[1] * shape[2] + shape[3]))
    value = tf.random_uniform(shape, minval, maxval)
    return tf.Variable(value, name='Weights')


def _create_bias(shape):
    # Initialize with a slight positive bias to prevent ReLU neurons from dying.
    value = tf.constant(0.1, shape=shape)
    return tf.Variable(value, name='Bias')


def _create_conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


class DeepQNetwork():
    def __init__(self, sess, num_actions, state_shape):
        """Creates a deep Q network.

        Args:
            sess: The associated TensorFlow session.
            num_actions: Number of possible actions.
            state_shape: A vector with three values, representing the width, height and depth of
                input states. For example, the shape of 100x80 RGB images is [100, 80, 3].
        """

        self.sess = sess
        width = state_shape[0]
        height = state_shape[1]
        depth = state_shape[2]
        self.x = tf.placeholder(tf.float32, [None, width, height, depth], name='Input_States')

        with tf.name_scope('Convolutional_Layer_1'):
            W_conv1 = _create_conv2d_weights([8, 8, depth, 32])
            b_conv1 = _create_bias([32])
            h_conv1 = tf.nn.relu(_create_conv2d(self.x, W_conv1, stride=4) + b_conv1)

        with tf.name_scope('Convolutional_Layer_2'):
            W_conv2 = _create_conv2d_weights([4, 4, 32, 64])
            b_conv2 = _create_bias([64])
            h_conv2 = tf.nn.relu(_create_conv2d(h_conv1, W_conv2, stride=2) + b_conv2)

        with tf.name_scope('Convolutional_Layer_3'):
            W_conv3 = _create_conv2d_weights([3, 3, 64, 64])
            b_conv3 = _create_bias([64])
            h_conv3 = tf.nn.relu(_create_conv2d(h_conv2, W_conv3, stride=1) + b_conv3)

        # Flatten the output to feed it into the fully connected layer.
        post_conv_height = math.ceil((math.ceil((height - 7) / 4) - 3) / 2) - 2
        post_conv_width = math.ceil((math.ceil((width - 7) / 4) - 3) / 2) - 2
        num_params = post_conv_height * post_conv_width * 64
        h_flat = tf.reshape(h_conv3, [-1, num_params])

        with tf.name_scope('Fully_Connected_Layer'):
            W_fc = _create_fc_weights([num_params, 512])
            b_fc = _create_bias([512])
            h_fc = tf.nn.relu(tf.matmul(h_flat, W_fc) + b_fc)

        # Implement dropout.
        self.keep_prob = tf.placeholder(tf.float32, name='Keep_Prob')
        h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)

        with tf.name_scope('Output'):
            W_output = _create_fc_weights([512, num_actions])
            b_output = _create_bias([num_actions])
            h_output = tf.matmul(h_fc_drop, W_output) + b_output

        # Estimate the optimal action and its expected value.
        self.optimal_action = tf.argmax(h_output, 1, name='Optimal_Action')
        self.optimal_action_value = tf.reduce_max(h_output, 1, name='Optimal_Action_Value')

        # Estimate the value of the specified action.
        self.action = tf.placeholder(tf.uint8, name='Action')
        one_hot_action = tf.one_hot(self.action, num_actions)
        self.Q = tf.reduce_sum(h_output * one_hot_action, 1, name='Estimated_Action_Value')

        # Compare with the observed action value.
        self.Q_ = tf.placeholder(tf.float32, [None], name='Observed_Action_Value')
        self.loss = tf.reduce_mean(tf.square(self.Q - self.Q_))
        self.learning_rate = tf.placeholder(tf.float32, name='Learning_Rate')
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def eval_optimal_action(self, state):
        """Estimates the optimal action for the specified state.

        Args:
            state: A state. Can be batched into multiple states.
        """

        return self.sess.run(self.optimal_action, feed_dict={self.x: state, self.keep_prob: 1})

    def eval_optimal_action_value(self, state):
        """Estimates the optimal action value for the specified state.

        Args:
            state: A state. Can be batched into multiple states.
        """

        return self.sess.run(self.optimal_action_value, feed_dict={self.x: state,
                                                                   self.keep_prob: 1})

    def eval_Q(self, state, action):
        """Evaluates the utility of the specified action for the specified state.

        Args:
            state: A state. Can be batched into multiple states.
            action: An action. Can be batched into multiple actions.
        """

        return self.sess.run(self.Q, feed_dict={self.x: state,
                                                self.action: action,
                                                self.keep_prob: 1})

    def train(self, state, action, Q_, learning_rate, dropout_prob):
        """Learns by performing one step of gradient descent.

        Args:
            state: A state. Can be batched into multiple states.
            action: An action. Can be batched into multiple actions.
            Q_: An observed action value.
            learning_rate: The speed with which the network learns from new examples.
            dropout_prob: Likelihood of individual neurons from the fully connected layer becoming
                inactive.
        """

        self.sess.run(self.train_step, feed_dict={self.x: state,
                                                  self.action: action,
                                                  self.Q_: Q_,
                                                  self.learning_rate: learning_rate,
                                                  self.keep_prob: 1 - dropout_prob})
