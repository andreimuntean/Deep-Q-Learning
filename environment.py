"""Augments OpenAI Gym Atari environments with features like experience replay.

Heavily influenced by DeepMind's seminal paper 'Playing Atari with Deep Reinforcement Learning'
(Mnih et al., 2013).
"""

from collections import deque
from skimage import color
from skimage import transform

import numpy as np


def _preprocess_observation(observation):
    """Deletes colors, shrinks and crops the 210x160x3 observation into an 84x84 grayscale image."""

    smaller_image = transform.resize(color.rgb2gray(observation), (110, 84))
    square_image = smaller_image[17:110 - 9, :]

    # Convert values from 64-bit floats to 32-bit floats.
    return square_image.astype(np.float32)


def _get_state(observations):
    """Creates a state from the specified observations.

    States are series of consecutive observations. Using more than one observation per state may
    provide short-term memory for the learner. Great for games like Pong where the trajectory of the
    ball can't be inferred from a single image.

    Returns:
        An 84x84xlen(observations) tensor with values from 0 to 1.
    """

    state = np.empty([84, 84, len(observations)], np.float32)

    for i, observation in enumerate(observations):
        state[:, :, i] = observation

    return state


class AtariWrapper:
    """Wraps over an Atari environment from OpenAI Gym and provides experience replay."""

    def __init__(self,
                 env,
                 replay_memory_capacity=100000,
                 observations_per_state=3,
                 action_space=None):
        """Creates the wrapper.

        Args:
            env: An OpenAI Gym Atari environment.
            replay_memory_capacity: Number of experiences remembered. Conceptually, an experience is
                a (state, action, reward, next_state, done) tuple. During training, learners sample
                the replay memory.
            observations_per_state: Number of consecutive observations within a state. Provides some
                short-term memory for the learner. Useful in games like Pong where the trajectory of
                the ball can't be inferred from a single image.
            action_space: Determines which actions are allowed. If 'None', all actions are allowed.
        """

        self.env = env
        self.replay_memory_capacity = replay_memory_capacity
        self.observations_per_state = observations_per_state
        self.action_space = action_space if action_space else list(range(self.env.action_space.n))
        self.state_space = [84, 84, observations_per_state]
        self._initialize_replay_memory()
        self.restart()

    def restart(self):
        """Restarts the game."""

        self.env.reset()
        self.done = False
        
    def step(self, action):
        """Performs the specified action.

        Returns:
            The reward.
        
        Raises:
            Exception: If the game ended.
            ValueError: If the action is not valid.
        """

        if self.done:
            raise Exception('Game finished.')

        if action not in self.action_space:
            raise ValueError('Action "{}" is invalid. Valid actions: {}.'.format(action,
                                                                                 self.action_space))
        
        observation, reward, self.done, _ = self.env.step(action)

        self.info.append((action, reward, self.done))
        self.observations.append(_preprocess_observation(observation))

        return reward

    def render(self):
        """Draws the environment."""

        self.env.render()

    def sample_action(self):
        """Samples a random action."""

        return np.random.choice(self.action_space)

    def sample_experiences(self, exp_count):
        """Randomly samples experiences from the replay memory. May contain duplicates."""

        indexes = np.random.choice(len(self.info), exp_count)

        states = np.empty([exp_count, 84, 84, self.observations_per_state], np.float32)
        actions = np.empty(exp_count, np.uint8)
        rewards = np.empty(exp_count, np.float32)
        next_states = np.empty([exp_count, 84, 84, self.observations_per_state], np.float32)
        done = np.empty(exp_count, np.bool)

        for i, sampled_i in enumerate(indexes):
            # Initial state = [sampled_i, sampled_i + observations_per_state].
            # Next state = [sampled_i + 1, sampled_i + observations_per_state + 1].
            # So final range (their union) is [sampled_i, sampled_i + observations_per_state + 1].
            observation_i = range(sampled_i, sampled_i + self.observations_per_state + 1)
            observations = [self.observations[j] for j in observation_i]

            states[i] = _get_state(observations[:self.observations_per_state])
            actions[i] = self.info[sampled_i][0]
            rewards[i] = self.info[sampled_i][1]
            next_states[i] = _get_state(observations[1:])
            done[i] = self.info[sampled_i][2]

        return states, actions, rewards, next_states, done

    def get_state(self):
        """Gets the current state.

        Returns:
            An 84x84x(self.observations_per_state) tensor with values from 0 to 1.
        """

        observations = [self.observations[i] for i in range(-self.observations_per_state, 0)]
        current_state = _get_state(observations)

        return current_state

    def _initialize_replay_memory(self):
        """Clears the experience buffer then creates the initial experience by acting randomly."""

        self.observations = deque(maxlen=self.replay_memory_capacity + self.observations_per_state)
        self.info = deque(maxlen=self.replay_memory_capacity)

        # Prepare the first state by performing random actions. States are represented by
        # observations_per_state consecutive observations.
        for i in range(self.observations_per_state):
            observation, _, _, _ = self.env.step(self.sample_action())
            self.observations.append(_preprocess_observation(observation))

        # Prepare the next state by performing one more random action.
        action = self.sample_action()
        observation, reward, self.done, _ = self.env.step(action)

        # Store the first experience.
        self.info.append((action, reward, self.done))
        self.observations.append(_preprocess_observation(observation))
