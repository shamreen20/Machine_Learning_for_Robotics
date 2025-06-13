# Deep Q-Learning with Ray RLlib for CartPole and LunarLander
This repository contains a Jupyter notebook (DQN.ipynb) that demonstrates the implementation of Deep Q-Learning (DQL) using the Ray RLlib framework to solve the CartPole-v1 and LunarLander-v2 environments from Gymnasium. The notebook explores advanced reinforcement learning techniques, including the Rainbow DQN algorithm, which integrates enhancements like Double DQN, Dueling DQN, Noisy Networks, and Distributional Q-Learning.

## Overview
The notebook focuses on applying DQL to environments with continuous state spaces, where traditional tabular Q-Learning is infeasible. It leverages Ray RLlib to implement the Rainbow algorithm, a state-of-the-art DQN variant that combines multiple improvements for better performance and stability. The notebook includes:

1. **Setup and Dependencies:** Installation of required packages (``gymnasium[box2d]``, ``ray[rllib]``, ``ray[tune]``, ``mediapy``) and system dependencies (swig, build-essential, python3-dev) for running simulations.

2. **DQN Implementation:** Configuration of the Rainbow DQN algorithm using ``DQNConfig`` with parameters like learning rate, multi-step learning, noisy networks, double Q-learning, dueling architecture, and distributional Q-values.

3. **Training and Evaluation:** Training on the CartPole-v1 environment with a target mean episode reward of >450, and an attempt to solve LunarLander-v2 with a target mean reward of >200.

4. **Metric Tracking:** Instructions for using TensorBoard to monitor metrics like episode reward mean, Q-loss, exploration parameters, and policy entropy, logged to ``/root/ray_results``.

5. **Hyperparameter Tuning:** Experimentation with DQN configurations to optimize performance, addressing challenges like overestimation bias and training stability.

## Key Concepts

- **Deep Q-Learning (DQL):** Extends Q-Learning by using a neural network to approximate Q-values for continuous state spaces, enabling scalability to complex environments.
- **Rainbow DQN:** Combines enhancements like:
    - **Double DQN:** Reduces overestimation bias in Q-value updates.
    - **Dueling DQN:** Separates state value and action advantage estimation for better learning efficiency.
    - **Noisy Networks:** Introduces parametric noise for exploration, replacing traditional epsilon-greedy strategies.
    - **Distributional Q-Learning:** Models the distribution of Q-values for more robust value estimation.
    - **Experience Replay:** Stores and samples past experiences to improve data efficiency and break sample correlations.
    - **Target Network:** Stabilizes training Ascent Gradient Descent by using a separate network for target Q-values.


- **CartPole-v1:** A classic control problem with a continuous 4D state space (position, velocity, pole angle, angular velocity) and discrete actions (push left or right). The goal is to balance the pole for as long as possible, targeting a mean reward of >450.
- **LunarLander-v2:** A more complex environment with an 8D state space and sparse rewards, requiring a mean reward of >200 to be considered solved.

## Notebook Structure

1. **Installation Cells:** Commands to install necessary dependencies for running the notebook in a Colab environment.
2. **DQN Configuration for CartPole:** Configures a Rainbow DQN with tuned hyperparameters (e.g., learning rate of 1e-3, deeper network with [64, 64] hidden layers) and trains until achieving a mean reward of >450.
3. **Policy Evaluation:** Uses the trained policy to render an episode of CartPole-v1.
4. **Metric Tracking with TensorBoard:** Instructions to visualize training metrics like episode reward mean, Q-loss, and exploration parameters.
5. **LunarLander Configuration:** Adjusts the DQN configuration (e.g., larger network [128, 128], adjusted reward range) to tackle the more challenging LunarLander-v2 environment.

## Challenges and Solutions

- **Continuous State Space:** Addressed by using a neural network to approximate Q-values, avoiding the need for discretizing the state space.
- **Training Stability:** Enhanced by Rainbow DQN features like double Q-learning, dueling architecture, and noisy networks, which improve exploration and reduce overestimation bias.
- **Hyperparameter Sensitivity:** The notebook experiments with learning rates, network architectures, and multi-step learning to achieve robust performance.
- **Error Handling:** Notes deprecated methods (e.g., config.rollouts) and suggests using AlgorithmConfig.env_runners for compatibility with newer Ray RLlib versions.

## Running the Notebook

1. **Environment Setup:** Ensure a Colab environment with GPU support (e.g., T4) for faster training.
2. **Install Dependencies:** Run the installation cells to set up swig, box2d-py, gymnasium, ray, and mediapy.
3. **Launch TensorBoard:** Use %tensorboard --logdir /root/ray_results to monitor training metrics in real-time.
4. **Execute Training Cells:** Run the DQN training cells for CartPole-v1 and optionally LunarLander-v2, adjusting hyperparameters as needed.
5. **Visualize Results:** Use the rendering cell to visualize the trained policy’s performance.

## Notes

- The notebook was tested in a Colab environment with Python 3.11.9 and a T4 GPU.
- Training for CartPole-v1 typically converges in <500 episodes, while LunarLander-v2 may require 1,000–2,000 episodes due to its complexity.
- Check /root/ray_results for training logs and use TensorBoard to explore metrics like episode reward mean, Q-loss, and policy entropy.
- For LunarLander-v2, ensure the reward range (v_min, v_max) accounts for negative rewards (e.g., crashes).

## Work

- Experiment with additional hyperparameter tuning (e.g., learning rate schedules, exploration decay) to further optimize performance.
- Test the configuration on other Gymnasium environments to evaluate generalization.
- Address deprecated Ray RLlib methods by updating to AlgorithmConfig.env_runners and re-testing.

This notebook provides a practical introduction to advanced DQL techniques and demonstrates how frameworks like Ray RLlib simplify the implementation of state-of-the-art reinforcement learning algorithms.
