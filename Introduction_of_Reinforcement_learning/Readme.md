# Grid-World Markov Decision Process (MDP) Implementation
This project implements a discrete Markov Decision Process (MDP) in a 4x4 grid-world environment using Python, Gymnasium, and Pygame. The agent (blue circle) aims to reach a target (red square) in the least amount of time to maximize its reward. The implementation includes a custom Gymnasium environment, dynamic programming for computing optimal policies and value functions, Monte Carlo methods for learning from sampled data, and Q-learning for sample-based policy optimization.

## Overview
The MDP is formalized as ⟨S, A, R, P, γ⟩, where:

- **S:** Set of 15 discrete states (4x4 grid minus the goal state).
- **A:** Four actions (move right, up, left, down).
- **R:** Reward function, providing a reward of 1 when the agent reaches the target and 0 otherwise.
- **P:** Deterministic transition dynamics, where the agent moves in the chosen direction unless blocked by grid boundaries.
- **γ:** Discount factor (0 ≤ γ < 1), encouraging the agent to reach the goal quickly.

The agent interacts with the environment in discrete time steps, choosing an action, receiving a reward, and transitioning to a new state. The episode terminates when the agent reaches the target.

## Environment
The grid-world is implemented as a Gymnasium environment (GridWorldEnv) with the following components:

- **Observation Space:** A dictionary containing the agent's and target's locations as (x, y) coordinates in the 4x4 grid.
- **Action Space:** Discrete space with four actions (0: right, 1: up, 2: left, 3: down).
- **Key Methods:**
    - ``reset(seed=None, target=None)``: Initializes the agent's and target's positions, ensuring they do not coincide.
    - ``step(action)``: Executes an action, updates the agent's position, returns the new observation, reward, termination status, and additional info (e.g., Manhattan distance to the target).
    - ``render(action=None)``: Visualizes the grid, agent (blue circle), target (red square), and action direction (if provided) using Pygame.


## Policies and Value Functions
**Policies**
A policy π(a|s) defines the probability of taking action a in state s. Policies can be stochastic or deterministic. The notebook starts with a uniform random policy (π(a|s) = 1/|A| = 0.25 for all actions) and progresses to an ε-greedy Q-policy for learning.
**State-Value Function**
The state-value function Vπ(s) represents the expected cumulative discounted reward starting from state s under policy π, defined recursively as:
````
Vπ(s) = E_a∼π(s) [ R(s,a) + γ E_s′∼P(s,a) [ Vπ(s′) ] ]
````

**Optimal Policy and Value Functions**
The goal is to find an optimal policy π* that maximizes the expected return, satisfying the Bellman optimality equation:
````
V*(s) = max_a∈A [ R(s,a) + γ Σ_s′∈S P(s,a)(s′) V*(s′) ]
````

The state-action value function Q*(s,a) is:
````
Q*(s,a) = R(s,a) + γ E_s′∼P(s,a) [ V*(s′) ]
````

The optimal policy is derived as:
````
π*(s) = argmax_a∈A Q*(s,a)
````

## Methods
**Dynamic Programming**
Value iteration is used to compute V* and Q* by iteratively applying the Bellman optimality equation, assuming full knowledge of R and P. This method is effective for environments with known dynamics.

**Monte Carlo Learning**
Monte Carlo methods estimate Vπ(s) and Qπ(s,a) by sampling full episodes and averaging returns. The state-value function is approximated as:
Vπ(s) ≈ (1/N(s)) Σ_i=1^N(s) G_i

where G_i is the discounted return from state s. An ε-greedy Q-policy is used, updated with a learning rate α:
````
V(s) ← (1−α)⋅V(s) + α⋅G_i
````

This approach is model-free, relying on interaction data without requiring knowledge of R or P.

**Q-Learning**
Q-learning combines dynamic programming's iterative refinement with Monte Carlo's sample-based learning. It updates the Q-values using:
````
Q(s_t, a_t) ← Q(s_t, a_t) + α [ r_{t+1} + γ max_a Q(s_{t+1}, a) - Q(s_t, a_t) ]
````

This update is policy-agnostic, allowing learning of the optimal policy from data collected by any policy that sufficiently explores the state-action space.

## Installation
To run the notebook, install the required dependencies:
````
pip install gymnasium pygame mediapy numpy matplotlib
````
Ensure Python 3.10 or later is installed. The notebook was tested with the ml4robotics kernel.

## Usage

1. **Run the Notebook:** Open ``E04_Intro_to_Reinforcement_Learning.ipynb`` in Jupyter Notebook or a compatible environment (e.g., Google Colab with GPU support).

2. **Install Dependencies:** Execute the first cell to install ``gymnasium``, ``pygame``,`` mediapy``, ``numpy``, and ``matplotlib``.

3. **Environment Setup:** The ``GridWorldEnv`` class defines the grid-world MDP. Run the relevant cells to initialize and test the environment.

4. **Visualization:** The notebook includes a cell that runs an episode with a random policy, rendering the agent's movements and displaying a video using ``mediapy``.

5. **Policy and Learning Experiments:**

Use value iteration to compute optimal policies and value functions.
Run Monte Carlo simulations to estimate value functions from sampled episodes.
Implement Q-learning to learn the optimal policy from interaction data.


## Key Features

- **Custom Gymnasium Environment:** Implements a 4x4 grid-world with deterministic transitions and sparse rewards.
- **Pygame Visualization:** Displays the agent, target, and grid with optional action arrows.
- **Multiple Learning Methods:** Supports dynamic programming, Monte Carlo, and Q-learning for policy optimization.
- **Episode Simulation:** Demonstrates a random policy with video output showing the agent's path.
- **Extensible Framework:** Easily modifiable for experimenting with different policies, rewards, or grid sizes.

## Code Structure

- **Imports:** Includes ``numpy``, ``pygame``, ``gymnasium``, ``mediapy``, ``matplotlib``, and ``scipy.ndimage`` for environment creation, visualization, and utilities.
- **GridWorldEnv Class:**
    - Defines the MDP's state space, action space, transitions, and rewards.
    - Handles rendering with Pygame, drawing the grid, agent, and target.


- Example Execution: Runs an episode with a random policy, collecting observations, rewards, and rendering frames, displayed as a video.

## Future Work

- Extend to stochastic transitions or environments with obstacles.
- Implement additional reinforcement learning algorithms (e.g., SARSA, policy gradients).
- Experiment with larger grids or continuous state/action spaces.
- Explore partial observability for more complex scenarios.
- Enhance visualization with policy or value function heatmaps.

## Notes

- The notebook assumes deterministic transitions; the agent moves as intended unless clipped by grid boundaries.
- The reward is sparse (1 at the goal, 0 elsewhere), encouraging the shortest path to the target.
- Monte Carlo and Q-learning methods highlight the model-free approach, applicable to unknown environments.
- The visualization uses Pygame for rendering and ``mediapy`` for video output, requiring a compatible environment.
