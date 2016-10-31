import environment
import gym

env = environment.AtariWrapper(gym.make('Pong-v0'), replay_memory_capacity=10)

print(env.action_space)
print(env.replay_memory_capacity)
print(env.replay_memory)
print()

env.start()

print(len(env.replay_memory))
print(env.replay_memory[0][0].shape)
print(env.replay_memory[0][1])
print(env.replay_memory[0][2])
print(env.replay_memory[0][3].shape)
print()

for i in range(15):
    env.step(env.sample_action())
    print(len(env.replay_memory))
