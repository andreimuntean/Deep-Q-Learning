"""Installs the modules required to run train_agent.py and test_agent.py."""

from setuptools import setup


setup(
    name='Deep Q-Network',
    version='1.0.0',
    url='https://github.com/andreimuntean/Deep-Q-Learning',
    description='Deep reinforcement learning using a deep Q-network with a dueling architecture.',
    author='Andrei Muntean',
    keywords='deep learning machine reinforcement neural network q-network dqn openai',
    install_requires=['gym[atari]', 'numpy', 'pillow', 'scipy', 'tensorflow']
)
