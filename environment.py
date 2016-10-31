"""Augments OpenAI Gym Atari environments with features like experience replay.

Heavily influenced by DeepMind's seminal paper 'Playing Atari with Deep Reinforcement Learning'
(Mnih et al., 2013).
"""

from collections import deque
from skimage import color
from skimage import transform

import numpy as np


def _preprocess_observation(observation):
    """Deletes color information, shrinks and crops the observation into an 84x84 image."""

    smaller_image = transform.resize(color.rgb2gray(observation), (110, 84))
    square_image = smaller_image[13:110 - 13, :]

    return square_image


def _get_next_state(state, observation):
    """Creates the next state from the current state and the specified observation.

    States are series of consecutive observations. Using more than one observation per state may
    provide short-term memory for the learner. Great for games like Pong where the trajectory of the
    ball can't be inferred from a single image.

    Returns:
        A len(state)x84x84 tensor."""

    next_state = np.empty(state.shape, dtype=state.dtype)
    
    # Get the past (self.observations_per_state - 1) observations.
    next_state[:-1] = state[1:]

    # Append the newest observation.
    next_state[-1] = observation

    return next_state


class AtariWrapper:
    """Wraps over an OpenAI Gym Atari environment and provides experience replay."""

    def __init__(self,
                 env,
                 replay_memory_capacity=100000,
                 observations_per_state=4,
                 action_space=None):
        """Creates the wrapper.

        Args:
            env: OpenAI Gym environment.
            replay_memory_capacity: Number of experiences remembered. An experience is a (state,
                action, reward, next_state) tuple. During training, learners sample the replay
                memory.
            observations_per_state: Number of consecutive observations within a state. Provides some
                short-term memory for the learner. Useful in games like Pong where the trajectory of
                the ball can't be inferred from a single image.
            action_space: Determines which actions are allowed. If none, all actions are allowed.
        """

        self.env = env
        self.replay_memory_capacity = replay_memory_capacity
        self.observations_per_state = observations_per_state
        self.action_space = action_space if action_space else list(range(self.env.action_space.n))
        self.replay_memory = deque()

    def start(self):
        """Starts (or restarts) the game."""

        self.replay_memory.clear()
        self.env.reset()

        # Construct the first state by performing random actions.
        state = np.empty([self.observations_per_state, 84, 84])

        for i in range(len(state)):
            observation, _, _, _ = self.env.step(self.sample_action())
            state[i] = _preprocess_observation(observation)

        # Construct the next state by performing another random action.
        action = self.sample_action()
        observation, reward, _, _ = self.env.step(action)
        next_state = _get_next_state(state, _preprocess_observation(observation))

        # Store the first experience into the replay memory.
        experience = state, action, reward, next_state
        self.replay_memory.append(experience)

    def step(self, action):
        """Performs the specified action."""
        
        state = self._get_state()
        observation, reward, done, _ = self.env.step(action)
        next_state = _get_next_state(state, _preprocess_observation(observation))
        experience = state, action, reward, next_state

        if len(self.replay_memory) >= self.replay_memory_capacity:
            self.replay_memory.popleft()
        
        self.replay_memory.append(experience)

    def sample_action(self):
        """Samples a random action."""

        return np.random.choice(self.action_space)

    def sample_experience(self, exp_count):
        """Randomly samples experiences from the replay memory. May contain duplicates."""

        return np.random.choice(self.replay_memory, exp_count)

    def _get_state(self):
        """Gets the current state.

        Returns:
            A (self.observations_per_state)x84x84 tensor."""

        return self.replay_memory[-1][0]