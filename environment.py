"""Augments OpenAI Gym Atari environments with features like experience replay.

Heavily influenced by DeepMind's seminal paper 'Playing Atari with Deep Reinforcement Learning'
(Mnih et al., 2013).
"""

from skimage import color
from skimage import transform

import gym
import numpy as np


# Specifies restricted action spaces. For games not in this dictionary, all actions are enabled.
ACTION_SPACE = {'Pong-v0': [0, 2, 3],  # NONE, UP and DOWN.
                'Breakout-v0': [1, 2, 3]}  # FIRE (respawn ball, otherwise NOOP), UP and DOWN.


class AtariWrapper:
    """Wraps over an Atari environment from OpenAI Gym and provides experience replay."""

    def __init__(self, env_name, replay_memory_capacity, observations_per_state, action_space=None):
        """Creates the wrapper.

        Args:
            env_name: Name of an OpenAI Gym Atari environment.
            replay_memory_capacity: Number of experiences remembered. Conceptually, an experience is
                a (state, action, reward, next_state, done) tuple. The replay memory is sampled by
                the agent during training.
            observations_per_state: Number of consecutive observations within a state. Provides some
                short-term memory for the learner. Useful in games like Pong where the trajectory of
                the ball can't be inferred from a single image.
            action_space: A list of possible actions. If 'action_space' is 'None' and no default
                configuration exists for this environment, all actions will be allowed.
        """

        self.env = gym.make(env_name)
        self.replay_memory_capacity = replay_memory_capacity
        self.state_length = observations_per_state
        self.done = False
        self.state_space = [84, 84, observations_per_state]

        if action_space:
            self.action_space = list(action_space)
        elif env_name in ACTION_SPACE:
            self.action_space = ACTION_SPACE[env_name]
        else:
            self.action_space = list(range(self.env.action_space.n))

        # Used when preprocessing observations.
        self.previous_frame = self.env.observation_space.low

        # Create replay memory. Arrays are used instead of double-ended queues for faster indexing.
        self.num_exp = 0
        self.actions = np.empty(replay_memory_capacity, np.uint8)
        self.rewards = np.empty(replay_memory_capacity, np.int8)
        self.ongoing = np.empty(replay_memory_capacity, np.bool)

        # Used for computing both 'current' and 'next' states.
        self.observations = np.empty([replay_memory_capacity + observations_per_state, 84, 84],
                                     np.float16)

        # Initialize the first state by performing random actions.
        for i in range(observations_per_state):
            observation, _, _, _ = self.env.step(self.sample_action())
            self.observations[i] = self._preprocess(observation)
            self.previous_frame = observation

        # Initialize the first experience by performing one more random action.
        self.step(self.sample_action())

    def reset(self):
        """Resets the environment."""

        self.env.reset()
        self.done = False

    def step(self, action):
        """Performs the specified action.

        Returns:
            The reward.

        Raises:
            ValueError: If the action is not valid.
        """

        if self.done:
            self.reset()

        if action not in self.action_space:
            raise ValueError('Action "{}" is invalid. Valid actions: {}.'.format(action,
                                                                                 self.action_space))

        observation, reward, self.done, _ = self.env.step(action)

        # Remember this experience.
        self.actions[self.num_exp] = action
        self.rewards[self.num_exp] = reward
        self.ongoing[self.num_exp] = not self.done
        self.observations[self.num_exp + self.state_length] = self._preprocess(observation)
        self.previous_frame = observation
        self.num_exp += 1

        if self.num_exp == self.replay_memory_capacity:
            # Free up space by deleting half of the oldest experiences.
            mid = int(self.num_exp / 2)
            end = 2 * mid

            self.num_exp = mid
            self.actions[:mid] = self.actions[mid:end]
            self.rewards[:mid] = self.rewards[mid:end]
            self.ongoing[:mid] = self.ongoing[mid:end]
            self.observations[:mid + self.state_length] = self.observations[mid:
                                                                            end + self.state_length]

        return reward

    def render(self):
        """Draws the environment."""

        self.env.render()

    def sample_action(self):
        """Samples a random action."""

        return np.random.choice(self.action_space)

    def sample_experiences(self, exp_count):
        """Randomly samples experiences from the replay memory. May contain duplicates.

        Args:
            exp_count: Number of experiences to sample.

        Returns:
            A (states, actions, rewards, next_states, ongoing) tuple. The boolean array, 'ongoing',
            determines whether the 'next_states' are terminal states.
        """

        indexes = np.random.choice(self.num_exp, exp_count)
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        ongoing = self.ongoing[indexes]
        states = np.array([self._get_state(i) for i in indexes])
        next_states = np.array([self._get_state(i) for i in indexes + 1])

        return states, actions, rewards, next_states, ongoing

    def get_state(self):
        """Gets the current state.

        Returns:
            An 84x84x(self.observations_per_state) tensor with real values between 0 and 1.
        """

        return self._get_state(-1)

    def _get_state(self, index):
        """Gets the specified state. Supports negative indexing.

        States are series of consecutive observations. Using more than one observation per state may
        provide short-term memory for the learner. Great for games like Pong where the trajectory of
        the ball can't be inferred from a single image.

        Returns:
            An 84x84x(observations_per_state) tensor with real values between 0 and 1.
        """

        state = np.empty([84, 84, self.state_length], np.float16)

        # Allow negative indexing by wrapping around.
        index = index % self.num_exp

        for i in range(self.state_length):
            state[..., i] = self.observations[index + i]

        return state

    def _preprocess(self, observation):
        """Transforms the specified observation into an 84x84 grayscale image.

        Returns:
            An 84x84 tensor with real values between 0 and 1.
        """

        # Create an intermediary frame by selecting the highest RGB values between this observation
        # and the previous one.
        intermediary_frame = np.maximum(self.previous_frame, observation)
        grayscale_image = color.rgb2gray(intermediary_frame)
        resized_image = transform.resize(grayscale_image, [84, 84])

        # Convert pixels from 64-bit floats (between 0 and 1) to 16-bit floats.
        return resized_image.astype(np.float16)
