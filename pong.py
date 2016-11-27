import agent
import argparse
import datetime
import environment
import gym
import tensorflow as tf

parser = argparse.ArgumentParser(description='Train an agent to play Pong.')

parser.add_argument('--load_path',
                    metavar='PATH',
                    help='loads a trained model from the specified path')

parser.add_argument('--save_path',
                    metavar='PATH',
                    help='saves the model at the specified path',
                    default='checkpoints/tmp/model.ckpt')

parser.add_argument('--save_interval',
                    metavar='TIME_STEPS',
                    help='time step interval at which to save the model',
                    type=int,
                    default=10000)

parser.add_argument('--num_episodes',
                    metavar='EPISODES',
                    help='number of episodes that will be played',
                    type=int,
                    default=1000)

parser.add_argument('--start_epsilon',
                    metavar='PERCENTAGE',
                    help='initial value for epsilon (exploration chance)',
                    type=float,
                    default=1)

parser.add_argument('--end_epsilon',
                    metavar='PERCENTAGE',
                    help='final value for epsilon (exploration chance)',
                    type=float,
                    default=0.05)

parser.add_argument('--anneal_duration',
                    metavar='EPISODES',
                    help='number of episodes to decrease epsilon from start_epsilon to end_epsilon',
                    type=int,
                    default=300)

parser.add_argument('--wait_before_training',
                    metavar='TIME_STEPS',
                    help='number of experiences to accumulate before training starts',
                    type=int,
                    default=5000)

parser.add_argument('--train_interval',
                    metavar='TIME_STEPS',
                    help='number of experiences to accumulate before next round of training starts',
                    type=int,
                    default=4)

parser.add_argument('--batch_size',
                    metavar='N',
                    help='number of experiences sampled and trained on at once',
                    type=int,
                    default=32)

parser.add_argument('--learning_rate',
                    metavar='PERCENTAGE',
                    help='rate at which the network learns from new examples',
                    type=float,
                    default=1e-6)

parser.add_argument('--discount',
                    metavar='PERCENTAGE',
                    help='discount factor for future rewards',
                    type=float,
                    default=0.99)

parser.add_argument('--target_network_update_factor',
                    metavar='PERCENTAGE',
                    help='rate at which target Q-network values shift toward real Q-network values',
                    type=float,
                    default=0.001)

args = parser.parse_args()
env = environment.AtariWrapper(gym.make('Pong-v0'),
                               action_space=[0, 2, 3], # 'NONE', 'UP' and 'DOWN'.
                               observations_per_state=4,
                               replay_memory_capacity=100000)

with tf.Session() as sess:
    learner = agent.Agent(sess,
                          env,
                          args.start_epsilon,
                          args.end_epsilon,
                          args.anneal_duration,
                          args.wait_before_training,
                          args.train_interval,
                          args.batch_size,
                          args.discount,
                          args.target_network_update_factor)

    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    save_next = args.save_interval

    if args.load_path:
        saver.restore(sess, args.load_path)
        print('Model restored.')

    for i in range(1, args.num_episodes + 1):
        reward = learner.train(render=True, learning_rate=args.learning_rate)
        print('Episode: {}  Reward: {}  Epsilon: {}'.format(i, reward, learner.get_epsilon()))

        if args.save_path and learner.t > save_next or i == args.num_episodes:
            saver.save(sess, args.save_path)
            print('[{}] Saved model at "{}".'.format(datetime.datetime.now(), args.save_path))
            save_next += args.save_interval
