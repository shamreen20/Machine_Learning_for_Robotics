# Model Predictive Control and Model-Based Reinforcement Learning
## Overview
This repository provides a comprehensive implementation of Model Predictive Control (MPC) and Model-Based Reinforcement Learning (MBRL) using forward dynamics models for planning and control. The framework leverages the MuJoCo physics engine for simulation and PyTorch for learning dynamics models. It includes tools for training forward models, optimizing control sequences with the Cross-Entropy Method (CEM) planner, and comparing MBRL with model-free RL methods like DQN, REINFORCE, and Actor-Critic (AC). The codebase is designed for applications in robotics, autonomous vehicles, and other control systems requiring robust and data-efficient planning.

## Features
- **Forward Dynamics Models:** Learn to predict system states using supervised learning with PyTorch.
- **Model Predictive Control (MPC):** Optimize control inputs over a finite horizon, handling constraints and nonlinear dynamics using the CEM planner.
- **Cross-Entropy Method (CEM) Planner:** Sampling-based optimization for action sequences in MPC, effective for non-differentiable cost functions.
- **Model-Based Reinforcement Learning (MBRL):** Combine learned dynamics models with planning for efficient decision-making.
- **Model-Free RL Comparison:** Evaluate MBRL against model-free RL methods (DQN, REINFORCE, AC) in terms of wall-clock time and data efficiency.
- **MuJoCo Integration:** Simulate complex physical systems for training and evaluation.
- **Visualization:** Use ``mediapy`` and ``matplotlib`` for rendering simulations and plotting results.

## Performance Comparison: MBRL vs. Model-Free RL
Wall-Clock Time:
Model-Free RL (DQN, REINFORCE, AC): Requires less computation per step but demands extensive environment interactions, leading to longer overall training times.
Model-Based RL: Slower per step due to planning with the forward model but converges faster with fewer real-world interactions.
Data Efficiency:
Model-Based RL: More data-efficient by leveraging simulated rollouts from the learned model, reducing the need for real-world data.
Model-Free RL: Less data-efficient, requiring significantly more environment interactions to learn effective policies.

## Model Predictive Control (MPC)
MPC is an advanced control strategy that leverages a predictive model of system dynamics to optimize control inputs over a finite time horizon. Unlike traditional methods like Linear Quadratic Regulator (LQR), MPC excels at handling nonlinear systems, constraints, and dynamic operating conditions. Key features include:

- **Optimization over a horizon:** Predicts future system behavior to optimize control inputs.
- **Constraint handling:** Incorporates limits on states, inputs, and outputs.
- **Applications:** Robotics, process control, autonomous vehicles.

## Planning with a Forward Model
Planning involves optimizing sequences of future states and actions to achieve an objective. Two primary approaches are:

- **Gradient-based methods:** Use gradients to iteratively refine solutions, efficient but may get stuck in local optima.
- **Sampling-based methods:** Explore state-action spaces through random sampling, computationally intensive but better at avoiding local optima.

## CEM Planner
The ``Cross-Entropy Method (CEM)`` planner is a sampling-based optimization approach for MPC. It iteratively refines a distribution over action sequences by sampling, evaluating, and updating based on the best-performing sequences. Key advantages:

- Handles non-differentiable or multi-modal cost functions effectively.
- Implements receding horizon control by applying the first action of the best sequence at each step.

## Model-Based Reinforcement Learning (MBRL)
MBRL integrates reinforcement learning with predictive models of environment dynamics. By learning a model, MBRL enables agents to simulate scenarios and plan actions, offering:

- **Sample efficiency:** Uses simulated rollouts to reduce reliance on real-world data.
- **Task-agnostic models:** Learned dynamics can be reused across tasks.
- **Challenges:** Model inaccuracies and computational complexity in planning.

## Learning the Forward Model
A forward dynamics model predicts future states given current states and actions. It is trained using supervised learning on a dataset of state transitions and control inputs. The process involves:

1. Collecting data from system interactions under various control inputs.
2. Training a model (e.g., neural network) to minimize prediction error.
3. Using the model for planning in MBRL or MPC.

## Model-Based vs. Model-Free RL
The included notebook (``Model_Predictive_Control.ipynb``) compares Model-Based RL and Model-Free RL (e.g., DQN, REINFORCE, Actor-Critic) in terms of:

- **Wall-clock time:**
    - **Model-Free RL:** Requires less computation per step but needs extensive environment interactions, leading to longer training times.
    - **Model-Based RL:** Slower per step due to planning with the forward model but converges faster with fewer real-world interactions.


- Data-efficiency:
    - **Model-Based RL:** More data-efficient by leveraging simulated rollouts from the learned model, reducing the need for real-world data.
    - **Model-Free RL:** Less data-efficient, requiring significantly more environment interactions to learn effective policies.


## Getting Started
**Prerequisites**
- Python 3.10+
- GPU (optional, for faster training with PyTorch)
- Required libraries ( ``requirements.txt``):
    - ``swig``, ``mediapy``, ``mujoco==3.1.4``, ``torch``, ``torchrl``, ``gymnasium==0.28.1`` (with ``box2d`` and mujoco extras)
    - Additional dependencies: ``numpy``, ``matplotlib``, ``scipy``, ``glfw``, ``pyopengl``

## Installation
To run the code and explore the repository:

1. Clone the repository:
````
git clone https://github.com/your-repo/mpc-mbrl.git
````

2. Install dependencies:The notebook installs required packages, including:
````
swig, mediapy, mujoco==3.1.4, torch, torchrl, gymnasium==0.28.1 (with box2d and mujoco environments).
Ensure Python 3.10+ is installed.
Install dependencies using:pip install swig mediapy mujoco==3.1.4 torch torchrl 'gymnasium[box2d,mujoco]==0.28.1'
````
## Key Concepts
- **Forward Dynamics Models:** Predict future states given current states and actions, trained via supervised learning.
- **MPC with CEM Planner:** Optimizes action sequences over a finite horizon, robust to non-differentiable costs.
- **MBRL:** Uses learned models for planning, improving data efficiency over model-free RL.
- **MuJoCo:** Physics engine for simulating complex dynamics (e.g., robotic locomotion).
- **Data Efficiency:** MBRL reduces real-world interaction needs by simulating rollouts.
- **Wall-Clock Time:** MBRL may converge faster despite slower per-step planning.

## System Requirements

- Python: 3.10.14 (as specified in the notebook metadata)
- Hardware: GPU recommended (e.g., T4 for Colab compatibility)
- Dependencies: Listed in the notebook (see installation cell)

## Contributing
1. Fork the repository.
2. Create a feature branch (``git checkout -b feature/your-feature``).
3. Commit changes (``git commit -m 'Add your feature'``).
4. Push to the branch (``git push origin feature/your-feature``).
5. Open a pull request.

## Contact
For issues or questions, open a GitHub issue or contact the maintainers[shamreen.tabassum@mailbox.tu-dresden.de].
