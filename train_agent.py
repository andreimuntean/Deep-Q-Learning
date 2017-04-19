"""Trains an agent to play Atari games from OpenAI Gym.

Heavily influenced by DeepMind's seminal paper 'Playing Atari with Deep Reinforcement Learning'
(Mnih et al., 2013) and 'Human-level control through deep reinforcement learning' (Mnih et al.,
2015).
"""

import agent
import argparse
import environment
import logging
import os
import random
import tensorflow as tf


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


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

PARSER.add_argument('--log_dir',
                    metavar='PATH',
                    help='path to a directory where to save & restore the model and log events',
                    default='models/tmp')

PARSER.add_argument('--render',
                    help='determines whether to display the game screen of the agent',
                    dest='render',
                    action='store_true',
                    default=False)

PARSER.add_argument('--num_epochs',
                    metavar='EPOCHS',
                    help='number of epochs to train for',
                    type=int,
                    default=200)

PARSER.add_argument('--epoch_length',
                    metavar='TIME STEPS',
                    help='number of time steps per epoch',
                    type=int,
                    default=200000)

PARSER.add_argument('--test_length',
                    metavar='TIME STEPS',
                    help="number of time steps per test",
                    type=int,
                    default=50000)

PARSER.add_argument('--max_episode_length',
                    metavar='TIME STEPS',
                    help='maximum number of time steps per episode',
                    type=int,
                    default=50000)

PARSER.add_argument('--test_epsilon',
                    metavar='EPSILON',
                    help='fixed exploration chance used when testing the agent',
                    type=float,
                    default=0.001)

PARSER.add_argument('--start_epsilon',
                    metavar='EPSILON',
                    help='initial value for epsilon (exploration chance)',
                    type=float,
                    default=1)

PARSER.add_argument('--end_epsilon',
                    metavar='EPSILON',
                    help='final value for epsilon (exploration chance)',
                    type=float,
                    default=0.01)

PARSER.add_argument('--anneal_duration',
                    metavar='TIME STEPS',
                    help='number of time steps to anneal epsilon from start_epsilon to end_epsilon',
                    type=int,
                    default=1000000)

PARSER.add_argument('--replay_memory_capacity',
                    metavar='EXPERIENCES',
                    help='number of most recent experiences remembered',
                    type=int,
                    default=500000)

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

PARSER.add_argument('--target_network_reset_interval',
                    metavar='TAU',
                    help=('number of experiences to accumulate before target Q-network values '
                          'reset to real Q-network values'),
                    type=float,
                    default=10000)

PARSER.add_argument('--batch_size',
                    metavar='EXPERIENCES',
                    help='number of experiences sampled and trained on at once',
                    type=int,
                    default=32)

PARSER.add_argument('--learning_rate',
                    metavar='LAMBDA',
                    help='rate at which the network learns from new examples',
                    type=float,
                    default=1e-4)

PARSER.add_argument('--max_gradient_norm',
                    metavar='DELTA',
                    help='maximum value allowed for the L2-norms of gradients',
                    type=float,
                    default=10)

PARSER.add_argument('--discount',
                    metavar='GAMMA',
                    help='discount factor for future rewards',
                    type=float,
                    default=0.99)

PARSER.add_argument('--observations_per_state',
                    metavar='FRAMES',
                    help='number of consecutive frames within a state',
                    type=int,
                    default=3)

PARSER.add_argument('--gpu_memory_alloc',
                    metavar='PERCENTAGE',
                    help='determines how much GPU memory to allocate for the neural network',
                    type=float,
                    default=0.25)


def main(args):
    """Trains an agent to play Atari games."""

    env = environment.AtariWrapper(args.env_name,
                                   args.max_episode_length,
                                   args.replay_memory_capacity,
                                   args.observations_per_state,
                                   args.action_space)
    test_env = environment.AtariWrapper(args.env_name,
                                        args.max_episode_length,
                                        100 * args.observations_per_state,
                                        args.observations_per_state,
                                        args.action_space)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    checkpoint_dir = os.path.join(args.log_dir, 'checkpoint')
    summary_dir = os.path.join(args.log_dir, 'summary')
    summary_writer = tf.summary.FileWriter(summary_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_alloc

    with tf.Session(config=config) as sess:
        player = agent.Agent(env,
                             args.start_epsilon,
                             args.end_epsilon,
                             args.anneal_duration,
                             args.train_interval,
                             args.target_network_reset_interval,
                             args.batch_size,
                             args.learning_rate,
                             args.max_gradient_norm,
                             args.discount)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=args.num_epochs)

        if args.load_path:
            saver.restore(sess, args.load_path)
            LOGGER.info('Restored model from "%s".', args.load_path)

        LOGGER.info('Accumulating %d experiences before training...', args.wait_before_training)

        for _ in range(args.wait_before_training):
            env.step(env.sample_action())

        env.reset()
        LOGGER.info('Accumulated %d experiences.', args.wait_before_training)

        for epoch_i in range(args.num_epochs):
            for _ in range(args.epoch_length):
                player.train()

                if args.render:
                    env.render()

                if env.done:
                    LOGGER.info('Finished episode. Total reward: %d. Length: %d.',
                                env.episode_reward,
                                env.episode_length)

                    summary = tf.Summary()
                    summary.value.add(tag='training/episode_length',
                                      simple_value=env.episode_length)
                    summary.value.add(tag='training/episode_reward',
                                      simple_value=env.episode_reward)
                    summary.value.add(tag='training/fps', simple_value=env.fps)
                    summary.value.add(tag='training/epsilon', simple_value=player.epsilon)

                    total_time_steps = args.train_interval * player.global_step.eval()
                    summary_writer.add_summary(summary, total_time_steps)
                    summary_writer.flush()

            file_name = '{}.{:05d}-of-{:05d}'.format(args.env_name, epoch_i, args.num_epochs)
            model_path = os.path.join(checkpoint_dir, file_name)
            saver.save(sess, model_path)
            LOGGER.info('Saved model to "%s".', model_path)

            # Evaluate the model.
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

            while time_step < args.test_length:
                local_total_reward = 0
                local_total_Q = 0
                local_min_Q = 1e7
                local_max_Q = -1e7
                local_time_step = 0
                test_env.reset()

                while not test_env.done and time_step + local_time_step < args.test_length:
                    local_time_step += 1
                    state = test_env.get_state()

                    # Occasionally try a random action (explore).
                    if random.random() < args.test_epsilon:
                        action = test_env.sample_action()
                    else:
                        action = player.get_action(state)

                    # Cast NumPy scalar to float.
                    Q = float(player.dqn.get_optimal_action_value(state))

                    # Record statistics.
                    local_total_reward += test_env.step(action)
                    local_total_Q += Q
                    local_min_Q = min(local_min_Q, Q)
                    local_max_Q = max(local_max_Q, Q)

                if not test_env.done:
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
            if num_games_finished > 0:
                # Extract more statistics.
                avg_reward = total_reward / num_games_finished
                avg_Q = total_Q / time_step
                avg_min_Q = summed_min_Qs / num_games_finished
                avg_max_Q = summed_max_Qs / num_games_finished

                summary = tf.Summary()
                summary.value.add(tag='testing/num_games_finished', simple_value=num_games_finished)
                summary.value.add(tag='testing/average_reward', simple_value=avg_reward)
                summary.value.add(tag='testing/minimum_reward', simple_value=min_reward)
                summary.value.add(tag='testing/maximum_reward', simple_value=max_reward)
                summary.value.add(tag='testing/average_Q', simple_value=avg_Q)
                summary.value.add(tag='testing/average_minimum_Q', simple_value=avg_min_Q)
                summary.value.add(tag='testing/minimum_Q', simple_value=min_Q)
                summary.value.add(tag='testing/average_maximum_Q', simple_value=avg_max_Q)
                summary.value.add(tag='testing/maximum_Q', simple_value=max_Q)

                summary_writer.add_summary(summary, epoch_i)
                summary_writer.flush()


if __name__ == '__main__':
    main(PARSER.parse_args())
