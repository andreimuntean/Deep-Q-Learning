"""Defines a deep Q network architecture.

Heavily influenced by DeepMind's seminal paper 'Playing Atari with Deep Reinforcement Learning'
(Mnih et al., 2013).
"""

import tensorflow as tf


def _create_weights(shape):
    # Initialize with slight noise to avoid symmetry.
    value = tf.truncated_normal(shape, stddev=0.1, name='weights')
    return tf.Variable(value)


def _create_bias(shape):
    # Initialize with a slight positive bias to prevent ReLU neurons from dying.
    value = tf.constant(0.1, shape=shape, name='bias')
    return tf.Variable(value)


def _create_conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def _create_max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


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
            W_conv1 = _create_weights([5, 5, depth, 32])
            b_conv1 = _create_bias([32])
            h_conv1 = tf.nn.relu(_create_conv2d(self.x, W_conv1) + b_conv1)

        with tf.name_scope('Max_Pool_1'):
            # Reshape [n, width, height, 32] output to [n, width/2, height/2, 32].
            h_pool1 = _create_max_pool_2x2(h_conv1)

        with tf.name_scope('Convolutional_Layer_2'):
            W_conv2 = _create_weights([5, 5, 32, 64])
            b_conv2 = _create_bias([64])
            h_conv2 = tf.nn.relu(_create_conv2d(h_pool1, W_conv2) + b_conv2)

        with tf.name_scope('Max_Pool_2'):
            # Reshape [n, width/2, height/2, 64] output to [n, width/4, height/4, 64].
            h_pool2 = _create_max_pool_2x2(h_conv2)

        # Flatten the [n, width/4, height/4, 64] output.
        h_pool2_flat = tf.reshape(h_pool2, [-1, width * height * 4])

        with tf.name_scope('Fully_Connected_Layer'):
            W_fc = _create_weights([width * height * 4, 256])
            b_fc = _create_bias([256])
            h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)

        # Implement dropout.
        self.keep_prob = tf.placeholder(tf.float32, name='Keep_Prob')
        h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)

        with tf.name_scope('Output'):
            W_output = _create_weights([256, num_actions])
            b_output = _create_bias([num_actions])
            h_output = tf.matmul(h_fc_drop, W_output) + b_output

        # Estimate the optimal action and its expected value.
        self.optimal_action = tf.argmax(h_output, 1, name='Optimal_Action')
        self.optimal_action_value = tf.reduce_max(h_output, 1, name='Optimal_Action_Value')

        # Estimate the value of the specified action.
        self.action = tf.placeholder(tf.int32, name='Action')
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

    def eval_loss(self, Q, Q_):
        """Compares a predicted Q value with an observed Q value."""
        
        return self.sess.run(self.loss, feed_dict={self.Q: Q, self.Q_: Q_})

    def train(self, state, action, Q_, learning_rate=0.001, dropout_prob=0.2):
        """Learns by performing one step of gradient descent.

        Args:
            state: A state. Can be batched into multiple states.
            action: An action. Can be batched into multiple actions.
            Q_: An observed action value.
            learning_rate: The speed with which the network learns from new examples.
            dropout_prob: Likelihood of individual neurons from fully connected layers becoming
                inactive.
        """
        
        self.sess.run(self.train_step, feed_dict={self.x: state,
                                                  self.action: action,
                                                  self.Q_: Q_,
                                                  self.learning_rate: learning_rate,
                                                  self.keep_prob: 1 - dropout_prob})
