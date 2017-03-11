# Deep Q-Learning
Deep reinforcement learning using a deep Q-network with a dueling architecture written in [TensorFlow](https://www.tensorflow.org/). 

This AI does not rely on hand-engineered rules or features. Instead, it masters the environment by looking at raw pixels and learning from experience, just as humans do.

## Dependencies
* OpenAI Gym 0.8
* TensorFlow 1.0

## Learning Environment
Uses environments provided by [OpenAI Gym](https://gym.openai.com/).

## Preprocessing
Each frame is transformed into a 48×48×3 image with 32-bit float values between 0 and 1. No image cropping is performed. Reward signals are restricted to -1, 0 and 1.

## Network Architecture
The input layer consists of a 48×48×3 image.

The first hidden layer convolves 64 filters of size 4×4 and stride 2, followed by a rectifier nonlinearity.

The second hidden layer convolves 64 filters of size 3×3 and stride 2, followed by another rectifier nonlinearity.

The third hidden layer convolves 64 filters of size 3×3 and stride 1, followed by another rectifier nonlinearity.

When using a dueling architecture, the network diverges into two streams – one computes the advantage of each possible action, the other the state value.

* The advantage stream consists of a fully-connected layer with 512 rectified linear units, feeding into as many output nodes as there are actions.

* The state value stream consists of a fully-connected layer with 512 rectified linear units, feeding into a single output node.

* The two streams merge and form the output layer. Each output node represents the expected utility of an action.

If a dueling architecture is not used:
* The last hidden layer consists of a fully-connected layer with 512 rectified linear units.
* The output layer has as many nodes as there are actions. Each output node represents the expected utility of an action.

## Acknowledgements
Heavily influenced by DeepMind's seminal paper ['Playing Atari with Deep Reinforcement Learning' (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602) and ['Human-level control through deep reinforcement learning' (Mnih et al., 2015)](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html).

Uses double Q-learning as described in [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461).

Uses the dueling architecture described in [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581).
