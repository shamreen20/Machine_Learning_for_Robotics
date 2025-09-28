# Policy Gradients Solution Notebook

This Jupyter notebook (`E06_PolicyGradients_solution.ipynb`) provides a practical implementation of policy gradient methods in reinforcement learning, focusing on solving the Hopper-v4 environment from Gymnasium's MuJoCo suite using Ray's RLlib library. It demonstrates the use of Proximal Policy Optimization (PPO) to train an agent, load a trained checkpoint, and render episodes. The notebook builds on concepts from basic policy gradients like REINFORCE to advanced actor-critic methods such as PPO and Soft Actor-Critic (SAC).

The notebook includes code for environment setup, training with PPO, and visualization of the trained agent's performance. It also outlines tasks for solving the Hopper environment and extending to other MuJoCo environments as homework.

## Key Components

### 1. Environment and Dependencies
The notebook uses the Hopper-v4 environment, a classic MuJoCo task where an agent learns to hop forward as far as possible. Dependencies are installed via pip, including:
- NumPy for numerical computations.
- MuJoCo for physics simulation.
- Ray (with RLlib and Tune) for distributed RL training.
- Gymnasium (with MuJoCo extras) for the environment.
- Mediapy for rendering videos.

### 2. Training with PPO
The core of the notebook involves configuring and running PPO using Ray RLlib. PPO is an advanced policy gradient algorithm that improves stability and sample efficiency over basic methods like REINFORCE by using a clipped surrogate objective.

Key steps:
- Import necessary libraries and define a rendering function (`render_episode`) to visualize agent trajectories.
- Set up the PPO trainer with custom configurations (e.g., environment, number of workers, learning rate).
- Train the agent using Ray Tune for hyperparameter tuning and checkpointing.
- Load the best checkpoint and compute actions for rendering.

Example training configuration (excerpt from notebook):
```python
config = (
    PPOConfig()
    .environment("Hopper-v4")
    .framework("torch")
    .rollouts(num_rollout_workers=4)
    .training(lr=0.0003, kl_coeff=0.2, train_batch_size=4000)
)
```

### 3. Rendering and Visualization
After training, the notebook loads the best policy from a checkpoint and renders an episode:
```python
best_policy = PPO.from_checkpoint(result.get_best_result().checkpoint).get_policy()
render_episode(lambda s: best_policy.compute_single_action(s, explore=False)[0], 'Hopper-v4')
```
This generates a video of the agent's behavior, embedded directly in the notebook output.

### 4. Tasks
- **Task 1**: Solve the Hopper-v4 environment using Ray RLlib's PPO or SAC. Refer to the [RLlib documentation](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html) for details.
- **Task 2 (Homework)**: Select and solve another MuJoCo environment from [Gymnasium MuJoCo](https://gymnasium.farama.org/environments/mujoco/), such as Walker2d-v4 or Ant-v4.

## Requirements
To run this notebook, ensure you have Python 3.11+ and install the dependencies:

```bash
pip install numpy==1.26.4 mujoco==2.3.2 ray==2.7 mediapy gymnasium[mujoco] ray[rllib] ray[tune]
```

Note: The notebook assumes a GPU-accelerated environment (e.g., Colab with T4 GPU) for faster training, but it can run on CPU.

## Usage
1. Clone or download the notebook.
2. Install dependencies as above.
3. Run the notebook in Jupyter or Colab.
4. Adjust hyperparameters in the PPO config for experimentation.
5. Monitor training progress via Ray Tune's logs and TensorBoard.

## Results
The trained PPO agent achieves stable hopping in Hopper-v4. The rendered video shows the agent maintaining balance and progressing forward. Training typically converges to a mean reward of ~3000+ after sufficient iterations.

For more on policy gradients:
- REINFORCE: Basic Monte-Carlo policy gradient.
- PPO: Clipped objective for stable updates (see [PPO paper](https://arxiv.org/abs/1707.06347)).
- SAC: Entropy-regularized actor-critic for continuous actions (see [SAC paper](https://arxiv.org/abs/1801.01290)).

If extending to other environments, modify the `environment` key in the config and ensure MuJoCo assets are properly loaded.

## License
This notebook is for educational purposes. Dependencies follow their respective licenses (e.g., Ray under Apache 2.0).