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

    return square_image


def _get_next_state(state, observation):
    """Creates the next state from the current state and the specified observation.

    States are series of consecutive observations. Using more than one observation per state may
    provide short-term memory for the learner. Great for games like Pong where the trajectory of the
    ball can't be inferred from a single image.

    Returns:
        A 84x84xlen(state) tensor.
    """

    next_state = np.empty(state.shape, dtype=state.dtype)
    
    # Get the past (self.observations_per_state - 1) observations.
    next_state[:, :, :-1] = state[:, :, 1:]

    # Append the newest observation.
    next_state[:, :, -1] = observation

    return next_state


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
            replay_memory_capacity: Number of experiences remembered. An experience is a [state,
                action, reward, next_state, done] array. During training, learners sample the replay
                memory.
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
        
        state = self.get_state()
        observation, reward, self.done, _ = self.env.step(action)
        next_state = _get_next_state(state, _preprocess_observation(observation))
        experience = np.array([state, action, reward, next_state, self.done])

        if len(self.replay_memory) >= self.replay_memory_capacity:
            self.replay_memory.popleft()
        
        self.replay_memory.append(experience)

        return reward

    def render(self):
        """Draws the environment."""

        self.env.render()

    def sample_action(self):
        """Samples a random action."""

        return np.random.choice(self.action_space)

    def sample_experiences(self, exp_count):
        """Randomly samples experiences from the replay memory. May contain duplicates."""

        indexes = np.random.choice(len(self.replay_memory), exp_count)
        experiences = np.array([self.replay_memory[i] for i in indexes])

        return experiences

    def get_state(self):
        """Gets the current state.

        Returns:
            A 84x84x(self.observations_per_state) tensor.
        """

        newest_experience = self.replay_memory[-1]
        current_state = newest_experience[3]

        return current_state

    def _initialize_replay_memory(self):
        """Clears the experience buffer then creates the initial experience by acting randomly."""

        self.replay_memory = deque()

        # Construct the first state by performing random actions.
        state = np.empty([84, 84, self.observations_per_state])

        for i in range(state.shape[2]):
            observation, _, _, _ = self.env.step(self.sample_action())
            state[:, :, i] = _preprocess_observation(observation)

        # Construct the next state by performing one more random action.
        action = self.sample_action()
        observation, reward, self.done, _ = self.env.step(action)
        next_state = _get_next_state(state, _preprocess_observation(observation))

        # Store the first experience into the replay memory.
        experience = np.array([state, action, reward, next_state, self.done])
        self.replay_memory.append(experience)