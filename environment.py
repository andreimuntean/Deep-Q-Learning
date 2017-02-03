"""Augments OpenAI Gym Atari environments with features like experience replay.

Heavily influenced by DeepMind's seminal paper 'Playing Atari with Deep Reinforcement Learning'
(Mnih et al., 2013) and 'Human-level control through deep reinforcement learning' (Mnih et al.,
2015).
"""

import cv2
import gym
import numpy as np


# Specifies restricted action spaces. For games not in this dictionary, all actions are enabled.
ACTION_SPACE = {'Pong-v0': [0, 2, 3],  # NONE, UP and DOWN.
                'Breakout-v0': [1, 2, 3]}  # FIRE (respawn ball, otherwise NOOP), UP and DOWN.


class AtariWrapper:
    """Wraps over an Atari environment from OpenAI Gym and provides experience replay."""

    def __init__(self,
                 env_name,
                 replay_memory_capacity,
                 observations_per_state,
                 frame_skip,
                 action_space=None):
        """Creates the wrapper.

        Args:
            env_name: Name of an OpenAI Gym Atari environment.
            replay_memory_capacity: Number of experiences remembered. Conceptually, an experience is
                a (state, action, reward, next_state, done) tuple. The replay memory is sampled by
                the agent during training.
            observations_per_state: Number of consecutive observations within a state. Provides some
                short-term memory for the learner. Useful in games like Pong where the trajectory of
                the ball can't be inferred from a single image.
            frame_skip: Number of frames per time step. Determines how many times an action selected
                by the agent is repeated.
            action_space: A list of possible actions. If 'action_space' is 'None' and no default
                configuration exists for this environment, all actions will be allowed.
        """

        self.env = gym.make(env_name)
        self.replay_memory_capacity = replay_memory_capacity
        self.state_length = observations_per_state
        self.frame_skip = frame_skip
        self.done = False
        self.state_space = [84, 84, observations_per_state]

        if action_space:
            self.action_space = list(action_space)
        elif env_name in ACTION_SPACE:
            self.action_space = ACTION_SPACE[env_name]
        else:
            self.action_space = list(range(self.env.action_space.n))

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
            self.observations[i], _ = self._step(self.sample_action())

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

        observation, reward = self._step(action)

        # Remember this experience.
        self.actions[self.num_exp] = action
        self.rewards[self.num_exp] = reward
        self.ongoing[self.num_exp] = not self.done
        self.observations[self.num_exp + self.state_length] = observation
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

    def _step(self, action):
        """Performs the specified action self.frame_skip times.

        Args:
            action: An action that will be repeated self.frame_skip times.

        Returns:
            An observation (84x84 tensor with real values between 0 and 1) and the accumulated
            reward.
        """

        frame, accumulated_reward, self.done, _ = self.env.step(action)
        accumulated_frame = frame.astype(np.int32)

        for _ in range(1, self.frame_skip):
            if self.done:
                break

            frame, reward, self.done, _ = self.env.step(action)
            accumulated_frame += frame
            accumulated_reward += reward

        # The environment may contain objects that flicker, becoming invisible to the agent every
        # few frames. To combat this, the past self.frame_skip frames are averaged into one.
        average_frame = accumulated_frame / self.frame_skip

        # Transform the average frame into a grayscale image with values between 0 and 1. Luminance
        # is extracted using the Y = 0.299 Red + 0.587 Green + 0.114 Blue formula. Values are scaled
        # between 0 and 1 by further dividing each color channel by 255.
        grayscale_frame = (average_frame[..., 0] * 0.00117 +
                           average_frame[..., 1] * 0.0023 +
                           average_frame[..., 2] * 0.00045)

        # Resize grayscale frame to an 84x84 matrix of 16-bit floats.
        observation = cv2.resize(
            grayscale_frame, (84, 84), interpolation=cv2.INTER_NEAREST).astype(np.float16)

        return observation, accumulated_reward
