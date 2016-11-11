#import dqn
#import tensorflow as tf
import environment
import gym

env = environment.AtariWrapper(gym.make('Pong-v0'),
                               action_space=[0, 2, 3], # 'NOOP', 'RIGHT' and 'LEFT'.
                               observations_per_state=2,
                               replay_memory_capacity=15000)

print('Possible actions:', env.action_space)
print('Replay memory capacity:', env.replay_memory_capacity)
print()

for i in range(20000):
    if env.done:
        print('Restarting...')
        env.restart()

    action = env.sample_action()
    env.step(action)
    print(action, len(env.replay_memory), env.observation_space)
    env.render()

input()

#sess = tf.InteractiveSession()
#dqn.DeepQNetwork(sess, len(env.action_space), env.observation_space)
#sess.close()