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
    def __init__(self, sess, num_actions, state_shape, learning_rate=0.001, dropout_prob=0.2):
        """Creates a deep Q network.

        Args:
            sess: The associated TensorFlow session.
            num_actions: Number of possible actions.
            state_shape: A vector with three values, representing the width, height and depth of
                input states. For example, the shape of 100x80 RGB images is [100, 80, 3].
            learning_rate: The speed with which the network learns from new examples.
            dropout_prob: Likelihood of individual neurons from fully connected layers becoming
                inactive.
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
        h_fc_drop = tf.nn.dropout(h_fc, dropout_prob)

        with tf.name_scope('Output'):
            W_output = _create_weights([256, num_actions])
            b_output = _create_bias([num_actions])
            self.Q = tf.matmul(h_fc_drop, W_output) + b_output

        self.Q_ = tf.placeholder(tf.float32, [None, num_actions], name='Observed_Q')
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.Q, self.Q_))
        self.train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def evaluate_Q(self, x):
        return self.sess.run(self.Q, feed_dict={self.x: x})

    def evaluate_loss(self, Q, Q_):
        return self.sess.run(self.loss, feed_dict={self.Q: Q, self.Q_: Q_})

    def train(self, states, Q_):
        self.sess.run(self.train_step, feed_dict={self.x: states, self.Q_: Q_})
