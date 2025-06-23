# Model-Based Reinforcement Learning (MBRL) Project
This repository contains a Jupyter Notebook (``E09_MBRL.ipynb``) implementing concepts of Model-Based Reinforcement Learning, including learning a forward dynamics model and policy distillation. Below is an overview of the key topics covered in the notebook.

##  Overview
The notebook demonstrates a practical implementation of MBRL using a Mujoco CartPole environment, showcasing how to learn environment dynamics, train a policy, and render episodes. It leverages libraries such as PyTorch, Mujoco, and others for efficient computation and visualization.

## Key Topics
**Model-Based Reinforcement Learning (MBRL)**
MBRL combines reinforcement learning with predictive modeling of environment dynamics. Unlike model-free RL, MBRL uses a learned model to simulate scenarios, enabling better planning and decision-making. This approach offers improved sample efficiency and task-agnosticism but faces challenges like model accuracy and computational complexity.

- **Key Features:**
    - Learns a model of environment dynamics for planning.
    - Simulates action outcomes to optimize decisions.
    - More sample-efficient than model-free RL.
    - Task-agnostic models allow reuse across tasks.
    - Challenges include model errors and computational cost.



## Learning a Forward Dynamics Model
The notebook implements a supervised learning approach to train a forward dynamics model. A neural network predicts future states based on current states and control inputs, minimizing prediction errors. This model is used for planning in the Mujoco CartPole task.

- **Process:**
    - Collect state-transition data under various control inputs.
    - Train a neural network to predict future states.
    - Use the trained model for simulation and planning in MBRL.



## Policy Distillation
Policy distillation transfers knowledge from a complex teacher policy" policy to a simpler "student" policy. The student policy is trained to mimic the teacher policy's behavior, achieving similar performance with lower computational demands.

- **Applications:**
    - Reduces computational complexity of teacher policies for real-time deployment.
    - Enables knowledge transfer to resource-constrained environments.
    - Maintains comparable performance with a simpler model.



## Notebook Contents

- **Environment Setup:**

    - Installs required libraries (``swig``, ``mediapy``, ``mujoco``, ``torch``, ``torchrl``, ``gymnasium``, etc.).
    - Configures the Mujoco CartPole environment with termination on failure.


- **Implementation:**

    - Trains a forward dynamics model for the CartPole environment.
    - Implements policy learning and evaluation.
    - Renders an episode using the trained policy, visualized in the notebook.


- **Dependencies:**

    - Python 3.10.14
    - Libraries: ``torch``, ``mujoco``, ``mediapy``, ``torchrl``, ``gymnasium``, etc.
    - GPU acceleration (e.g., NVIDIA T4 as used in Colab).



## Getting Started

1. **Clone the Repository:**
````
git clone <repository-url>
cd <repository-directory>
````

2. **Install Dependencies:**Run the notebook's installation cell to install required packages, or manually install them:
````
pip install swig mediapy mujoco==3.1.4 torch torchrl gymnasium==0.28.1
````

3. **Run the Notebook:**Open ``E09_MBRL.ipynb`` in Jupyter or Google Colab, and execute the cells to train the model and render episodes.


## Requirements

- Python 3.10+
- Jupyter Notebook or Google Colab
- GPU (optional, but recommended for faster training)

## Acknowledgments
- Built with inspiration from MBRL and policy distillation   research.
- Utilizes open-source libraries like PyTorch and Mujoco.
