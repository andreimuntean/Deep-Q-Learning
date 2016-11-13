import argparse
import datetime
import dqn
import environment
import gym
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Train an agent to play Pong.')

parser.add_argument('--load_path', help='loads a trained model from the specified path')

parser.add_argument('--save_path',
                    help='saves the model at the specified path',
                    default='checkpoints/tmp/model.ckpt')

parser.add_argument('--save_frequency',
                    help='time step interval at which to save the model',
                    type=int,
                    default=10000)

parser.add_argument('--num_episodes',
                    help='number of episodes that will be played',
                    type=int,
                    default=10000)

args = parser.parse_args()

load_path = args.load_path
save_path = args.save_path
save_frequency = args.save_frequency
num_episodes = args.num_episodes

batch_size = 32
wait_before_training = 5000
train_frequency = 3
discount = 0.99

env = environment.AtariWrapper(gym.make('Pong-v0'),
                               action_space=[0, 2, 3], # 'NONE', 'UP' and 'DOWN'.
                               observations_per_state=4,
                               replay_memory_capacity=20000)

epsilon_history = []
loss_history = []
reward_history = []

with tf.Session() as sess:
    network = dqn.DeepQNetwork(sess, len(env.action_space), env.state_space)
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    if load_path:
        saver.restore(sess, load_path)
        print('Model restored.')

    t = 0
    for i in range(1, num_episodes + 1):
        # Epsilon anneals from 1 to 0.05.
        epsilon = max(min(10 / i, 1), 0.05)
        episode_loss = []
        total_reward = 0

        while not env.done:
            t += 1
            env.render()

            # Occasionally train.
            if t > wait_before_training and t % train_frequency == 0:
                # These operations might be confusing if you forget that they're vectorized.
                experiences = env.sample_experiences(batch_size)
                states = np.stack(experiences[:, 0], axis=0)
                actions_i = np.stack([env.action_space.index(a) for a in experiences[:, 1]], axis=0)
                rewards = np.stack(experiences[:, 2], axis=0)
                next_states = np.stack(experiences[:, 3], axis=0)
                done = np.stack(experiences[:, 4], axis=0)

                # Estimate action values.
                Q = network.eval_Q(states, actions_i)

                # Determine the true Q values for the specified actions.
                #
                #                    { r, if next state is terminal
                # Q(state, action) = {
                #                    { r + discount * max(Q(next state, <any action>)), otherwise
                Q_ = rewards + ~done * discount * network.eval_optimal_action_value(next_states)
                
                # Estimate error and update weights.
                loss = network.eval_loss(Q, Q_)
                episode_loss.append(loss)
                network.train(states, actions_i, Q_)

            # Occasionally try a random action (explore).
            if np.random.rand() < epsilon:
                action = env.sample_action()
            else:
                state = np.expand_dims(env.get_state(), axis=0)
                action = env.action_space[network.eval_optimal_action(state)[0]]

            total_reward += env.step(action)

            if save_path and t % save_frequency == 0:
                saver.save(sess, save_path)
                print('[{}] Saved model at "{}".'.format(datetime.datetime.now(), save_path))

        epsilon_history.append(epsilon)
        loss_history.append(np.mean(episode_loss) if episode_loss else -1)
        reward_history.append(total_reward)
        print('Episode: {}  Loss: {}  Reward: {}  Epsilon: {}'.format(i,
                                                                      loss_history[-1],
                                                                      total_reward,
                                                                      epsilon))

        env.restart()

if save_path:
    saver.save(sess, save_path)
    print('[{}] Saved model at "{}".'.format(datetime.datetime.now(), save_path))

print('Total timesteps:', t)

plt.subplot(311)
plt.ylabel('Loss')
plt.xlabel('Episode')
plt.plot(loss_history)

plt.subplot(312)
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.plot(reward_history)

plt.subplot(313)
plt.ylabel('Explore / Exploit')
plt.xlabel('Episode')
plt.plot(epsilon_history)

plt.tight_layout()
plt.show()
