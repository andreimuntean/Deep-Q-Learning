"""Trains an agent to play Atari games from OpenAI gym.

Heavily influenced by DeepMind's seminal paper 'Playing Atari with Deep Reinforcement Learning'
(Mnih et al., 2013).
"""

import agent
import argparse
import csv
import datetime
import environment
import numpy as np
import os
import tensorflow as tf


PARSER = argparse.ArgumentParser(description='Train an agent to play Atari games.')

PARSER.add_argument('--env_name',
                    metavar='ENVIRONMENT',
                    help='name of an OpenAI Gym Atari environment on which to train',
                    default='Pong-v0')

PARSER.add_argument('--action_space',
                    nargs='+',
                    help='restricts the number of possible actions',
                    type=int)

PARSER.add_argument('--load_path',
                    metavar='PATH',
                    help='loads a trained model from the specified path')

PARSER.add_argument('--save_dir',
                    metavar='PATH',
                    help='saves the model at the specified path',
                    default='models/tmp')

PARSER.add_argument('--num_epochs',
                    metavar='EPOCHS',
                    help='number of epochs to train for',
                    type=int,
                    default=200)

PARSER.add_argument('--epoch_length',
                    metavar='TIME STEPS',
                    help='number of time steps per epoch',
                    type=int,
                    default=100000)

PARSER.add_argument('--num_tests',
                    metavar='TESTS',
                    help="number of tests after each epoch",
                    type=int,
                    default=200)

PARSER.add_argument('--start_epsilon',
                    metavar='EPSILON',
                    help='initial value for epsilon (exploration chance)',
                    type=float,
                    default=1)

PARSER.add_argument('--end_epsilon',
                    metavar='EPSILON',
                    help='final value for epsilon (exploration chance)',
                    type=float,
                    default=0.1)

PARSER.add_argument('--anneal_duration',
                    metavar='TIME STEPS',
                    help='number of time steps to anneal epsilon from start_epsilon to end_epsilon',
                    type=int,
                    default=1000000)

PARSER.add_argument('--replay_memory_capacity',
                    metavar='EXPERIENCES',
                    help='number of most recent experiences remembered',
                    type=int,
                    default=200000)

PARSER.add_argument('--wait_before_training',
                    metavar='TIME STEPS',
                    help='number of experiences to accumulate before training starts',
                    type=int,
                    default=50000)

PARSER.add_argument('--train_interval',
                    metavar='TIME STEPS',
                    help='number of experiences to accumulate before next round of training starts',
                    type=int,
                    default=4)

PARSER.add_argument('--batch_size',
                    metavar='EXPERIENCES',
                    help='number of experiences sampled and trained on at once',
                    type=int,
                    default=256)

PARSER.add_argument('--learning_rate',
                    metavar='LAMBDA',
                    help='rate at which the network learns from new examples',
                    type=float,
                    default=0.00025)

PARSER.add_argument('--dropout_prob',
                    metavar='DROPOUT',
                    help='likelihood of neurons from fully connected layers becoming inactive',
                    type=float,
                    default=0.7)

PARSER.add_argument('--discount',
                    metavar='GAMMA',
                    help='discount factor for future rewards',
                    type=float,
                    default=0.99)

PARSER.add_argument('--target_network_update_factor',
                    metavar='TAU',
                    help='rate at which target Q-network values shift toward real Q-network values',
                    type=float,
                    default=0.00005)

PARSER.add_argument('--observations_per_state',
                    metavar='FRAMES',
                    help='number of consecutive frames within a state',
                    type=int,
                    default=4)


def eval_model(env, player, num_tests, save_dir):
    """Evaluates the performance of the specified agent. Writes results in a CSV file."""

    avg_reward = 0
    avg_Q = 0
    avg_min_Q = 0
    min_min_Q = 1e7
    avg_max_Q = 0
    max_max_Q = -1e7

    for _ in range(num_tests):
        reward = 0
        total_Q = 0
        min_Q = 1e7
        max_Q = -1e7
        time_step = 0
        env.reset()

        while not env.done:
            state = env.get_state()
            action = player.get_action(state)
            Q = player.dqn.eval_optimal_action_value(np.expand_dims(state, axis=0))[0]
            time_step += 1

            # Record statistics.
            reward += env.step(action)
            total_Q += Q
            min_Q = min(min_Q, Q)
            max_Q = max(max_Q, Q)

        avg_reward += reward / num_tests
        avg_Q += total_Q / time_step / num_tests
        avg_min_Q += min_Q / num_tests
        avg_max_Q += max_Q / num_tests
        min_min_Q = min(min_min_Q, min_Q)
        max_max_Q = max(max_max_Q, max_Q)

    # Save results.
    with open(os.path.join(save_dir, 'log.csv'), 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([avg_reward, avg_Q, avg_min_Q, min_min_Q, avg_max_Q, max_max_Q])


def main(args):
    """Trains an agent to play Atari games."""

    env = environment.AtariWrapper(args.env_name,
                                   args.replay_memory_capacity,
                                   args.observations_per_state,
                                   args.action_space)

    with tf.Session() as sess:
        player = agent.Agent(sess,
                             env,
                             args.start_epsilon,
                             args.end_epsilon,
                             args.anneal_duration,
                             args.train_interval,
                             args.batch_size,
                             args.learning_rate,
                             args.dropout_prob,
                             args.discount,
                             args.target_network_update_factor)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=args.num_epochs)

        if args.load_path:
            saver.restore(sess, args.load_path)
            print('Restored model from "{}".'.format(args.load_path))

        # Accumulate experiences.
        for _ in range(args.wait_before_training):
            env.step(env.sample_action())

        for epoch_i in range(args.num_epochs):
            for _ in range(args.epoch_length):
                player.train()

            if args.save_dir:
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)

                file_name = '{}.{:05d}-of-{:05d}'.format(args.env_name, epoch_i, args.num_epochs)
                save_path = os.path.join(args.save_dir, file_name)
                saver.save(sess, save_path)
                print('[{}] Saved model to "{}".'.format(datetime.datetime.now(), save_path))

            eval_model(env, player, args.num_tests, args.save_dir)


if __name__ == '__main__':
    main(PARSER.parse_args())
