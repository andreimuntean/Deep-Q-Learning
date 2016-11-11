import dqn
import environment
import gym
import numpy as np
import sys
import tensorflow as tf

from matplotlib import pyplot as plt

env = environment.AtariWrapper(gym.make('Pong-v0'),
                               action_space=[0, 2, 3], # 'NOOP', 'RIGHT' and 'LEFT'.
                               observations_per_state=2,
                               replay_memory_capacity=15000)

print('Possible actions:', env.action_space)
print('Replay memory capacity:', env.replay_memory_capacity)
print('State space:', env.state_space)
print()

num_episodes = 20
batch_size = 32
wait_before_training = 5000
train_frequency = 4
discount = 0.99

load_path = None
save_path = '/tmp/model.ckpt'

epsilon_history = []
loss_history = []
reward_history = []

with tf.Session() as sess:
    network = dqn.DeepQNetwork(sess, len(env.action_space), env.state_space)
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    
    if load_path:
        saver.restore(sess, load_path)

    t = 0
    for i in range(1, num_episodes + 1):
        epsilon = min(10 / i, 1)
        episode_loss = []
        total_reward = 0

        while not env.done:
            t += 1
            env.render()
            
            # Occassionally train.
            if t > wait_before_training and t % train_frequency == 0:
                # These operations might be confusing if you forget that they're vectorized.
                experiences = env.sample_experiences(batch_size)
                states = np.stack(experiences[:, 0], axis=0)
                actions_i = np.stack([env.action_space.index(a) for a in experiences[:, 1]], axis=0)
                rewards = np.stack(experiences[:, 2], axis=0)
                next_states = np.stack(experiences[:, 3], axis=0)

                # Estimate the Q values.
                Q = network.evaluate_Q(states)

                # Get the true Q values for the specified actions.
                Q_ = np.copy(Q)
                observed_i = np.arange(batch_size), actions_i
                Q_[observed_i] = rewards + discount * np.max(network.evaluate_Q(next_states), 1)

                episode_loss.append(network.evaluate_loss(Q, Q_))
                network.train(states, Q_)

            # Occasionally try a random action (explore).
            if np.random.rand() < epsilon:
                action = env.sample_action()
            else:
                state = np.empty([1, *env.state_space])
                state[0] = env.get_state()
                strongest_signal = np.argmax(network.evaluate_Q(state))
                action = env.action_space[strongest_signal]

            total_reward += env.step(action)

        epsilon_history.append(epsilon)
        loss_history.append(np.mean(episode_loss))
        reward_history.append(total_reward)
        print('Episode: {}  Loss: {}  Reward: {}  Epsilon: {}'.format(i,
                                                                      loss_history[-1],
                                                                      total_reward,
                                                                      epsilon))

        if save_path and i % 10 == 0:
            saver.save(sess, save_path)
            print('Saved model.')

        env.restart()


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