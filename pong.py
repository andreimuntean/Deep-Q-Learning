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
print('State space:', env.state_space)
print()

for i in range(20000):
    if env.done:
        env.restart()
        print('Game restarted.')

    action = env.sample_action()
    env.step(action)
    env.render()

input()

#sess = tf.InteractiveSession()
#dqn.DeepQNetwork(sess, len(env.action_space), env.state_space)
#sess.close()