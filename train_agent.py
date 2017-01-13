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
import random
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
                    default=250)

PARSER.add_argument('--epoch_length',
                    metavar='TIME STEPS',
                    help='number of time steps per epoch',
                    type=int,
                    default=200000)

PARSER.add_argument('--test_length',
                    metavar='TIME STEPS',
                    help="number of time steps per test",
                    type=int,
                    default=75000)

PARSER.add_argument('--test_epsilon',
                    metavar='EPSILON',
                    help='fixed exploration chance used when testing the agent',
                    type=float,
                    default=0)

PARSER.add_argument('--start_epsilon',
                    metavar='EPSILON',
                    help='initial value for epsilon (exploration chance)',
                    type=float,
                    default=1)

PARSER.add_argument('--end_epsilon',
                    metavar='EPSILON',
                    help='final value for epsilon (exploration chance)',
                    type=float,
                    default=0.05)

PARSER.add_argument('--anneal_duration',
                    metavar='TIME STEPS',
                    help='number of time steps to anneal epsilon from start_epsilon to end_epsilon',
                    type=int,
                    default=2000000)

PARSER.add_argument('--replay_memory_capacity',
                    metavar='EXPERIENCES',
                    help='number of most recent experiences remembered',
                    type=int,
                    default=1000000)

PARSER.add_argument('--wait_before_training',
                    metavar='TIME STEPS',
                    help='number of experiences to accumulate before training starts',
                    type=int,
                    default=100000)

PARSER.add_argument('--train_interval',
                    metavar='TIME STEPS',
                    help='number of experiences to accumulate before next round of training starts',
                    type=int,
                    default=4)

PARSER.add_argument('--batch_size',
                    metavar='EXPERIENCES',
                    help='number of experiences sampled and trained on at once',
                    type=int,
                    default=32)

PARSER.add_argument('--learning_rate',
                    metavar='LAMBDA',
                    help='rate at which the network learns from new examples',
                    type=float,
                    default=0.0002)

PARSER.add_argument('--dropout_prob',
                    metavar='DROPOUT',
                    help='likelihood of neurons from fully connected layers becoming inactive',
                    type=float,
                    default=0.2)

PARSER.add_argument('--max_gradient',
                    metavar='DELTA',
                    help='maximum value allowed for gradients during backpropagation',
                    type=float,
                    default=10)

PARSER.add_argument('--discount',
                    metavar='GAMMA',
                    help='discount factor for future rewards',
                    type=float,
                    default=0.99)

PARSER.add_argument('--target_network_update_factor',
                    metavar='TAU',
                    help='rate at which target Q-network values shift toward real Q-network values',
                    type=float,
                    default=0.0004)

PARSER.add_argument('--observations_per_state',
                    metavar='FRAMES',
                    help='number of consecutive frames within a state',
                    type=int,
                    default=4)

PARSER.add_argument('--gpu_memory_alloc',
                    metavar='PERCENTAGE',
                    help='determines how much GPU memory to allocate for the neural network',
                    type=float,
                    default=0.2)


def eval_model(player, env, test_length, epsilon, save_path):
    """Evaluates the performance of the specified agent. Writes results in a CSV file.

    Args:
        player: An agent.
        env: Environment in which the agent is tested.
        test_length: Number of time steps to test the agent for.
        epsilon: Likelihood of the agent performing a random action.
        save_path: CSV file where results will be saved.
    """

    total_reward = 0
    min_reward = 1e7
    max_reward = -1e7
    total_Q = 0
    summed_min_Qs = 0
    min_Q = 1e7
    summed_max_Qs = 0
    max_Q = -1e7
    time_step = 0
    num_games_finished = 0

    while time_step < test_length:
        local_total_reward = 0
        local_total_Q = 0
        local_min_Q = 1e7
        local_max_Q = -1e7
        local_time_step = 0
        env.reset()

        while time_step + local_time_step < test_length and not env.done:
            local_time_step += 1
            state = env.get_state()

            # Occasionally try a random action (explore).
            if random.random() < epsilon:
                action = env.sample_action()
            else:
                action = player.get_action(state)

            Q = player.dqn.eval_optimal_action_value(np.expand_dims(state, axis=0))[0]

            # Record statistics.
            local_total_reward += env.step(action)
            local_total_Q += Q
            local_min_Q = min(local_min_Q, Q)
            local_max_Q = max(local_max_Q, Q)

        if not env.done:
            # Discard unfinished game.
            break

        num_games_finished += 1
        time_step += local_time_step
        total_reward += local_total_reward
        min_reward = min(min_reward, local_total_reward)
        max_reward = max(max_reward, local_total_reward)
        total_Q += local_total_Q
        summed_min_Qs += local_min_Q
        summed_max_Qs += local_max_Q
        min_Q = min(min_Q, local_min_Q)
        max_Q = max(max_Q, local_max_Q)

    # Save results.
    with open(save_path, 'a') as csvfile:
        writer = csv.writer(csvfile)

        if num_games_finished > 0:
            # Extract more statistics.
            avg_reward = total_reward / num_games_finished
            avg_Q = total_Q / time_step
            avg_min_Q = summed_min_Qs / num_games_finished
            avg_max_Q = summed_max_Qs / num_games_finished

            writer.writerow([num_games_finished,
                             avg_reward,
                             min_reward,
                             max_reward,
                             avg_Q,
                             avg_min_Q,
                             min_Q,
                             avg_max_Q,
                             max_Q])
        else:
            # The agent got stuck during the first game.
            writer.writerow([0, 0, 0, 0, 0, 0, 0, 0, 0])


def main(args):
    """Trains an agent to play Atari games."""

    env = environment.AtariWrapper(args.env_name,
                                   args.replay_memory_capacity,
                                   args.observations_per_state,
                                   args.action_space)
    test_env = environment.AtariWrapper(args.env_name,
                                        100 * args.observations_per_state,
                                        args.observations_per_state,
                                        args.action_space)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_alloc

    with tf.Session(config=config) as sess:
        player = agent.Agent(sess,
                             env,
                             args.start_epsilon,
                             args.end_epsilon,
                             args.anneal_duration,
                             args.train_interval,
                             args.batch_size,
                             args.learning_rate,
                             args.dropout_prob,
                             args.max_gradient,
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

        print('[{}] Accumulated {} experiences.'.format(
            datetime.datetime.now(), args.wait_before_training))

        for epoch_i in range(args.num_epochs):
            for _ in range(args.epoch_length):
                player.train()

            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

            file_name = '{}.{:05d}-of-{:05d}'.format(args.env_name, epoch_i, args.num_epochs)
            model_path = os.path.join(args.save_dir, file_name)
            saver.save(sess, model_path)
            print('[{}] Saved model to "{}".'.format(datetime.datetime.now(), model_path))

            results_path = os.path.join(args.save_dir, 'test_results.csv')
            eval_model(player, test_env, args.test_length, args.test_epsilon, results_path)
            print('[{}] Saved test results to "{}".'.format(datetime.datetime.now(), results_path))


if __name__ == '__main__':
    main(PARSER.parse_args())
