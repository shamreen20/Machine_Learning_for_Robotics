# Introduction to Control Systems with MuJoCo
This repository contains a Jupyter notebook (E03_Intro_to_Control1.ipynb) that provides an introduction to control systems using the MuJoCo physics engine. The notebook demonstrates the application of PID (Proportional-Integral-Derivative) and LQR (Linear Quadratic Regulator) control techniques to a 2D cartpole inverted pendulum model, a classic example in control theory. It also touches on the concept of model misspecification and its impact on controller performance.

## Overview
**MuJoCo**

MuJoCo (Multi-Joint dynamics with Contact) is a high-performance physics engine used in robotics, biomechanics, graphics, and machine learning research. Developed by Emo Todorov at the University of Washington, MuJoCo excels at simulating complex dynamic systems with articulated bodies and contact dynamics. Its key features include:

- Efficient computation of forward and inverse dynamics
- Accurate collision detection
- User-friendly interface and extensive documentation

For a comprehensive introduction, refer to the Google DeepMind tutorial and the official MuJoCo documentation.

**PID Control**

PID control is a feedback mechanism widely used in engineering to regulate system behavior. It combines three terms:

- **Proportional:** Responds to the current error (difference between setpoint and measured value)
- **Integral:** Accounts for past errors to eliminate steady-state error
- **Derivative:** Anticipates future errors based on the rate of change

The notebook allows users to experiment with tuning PID parameters for a cartpole system to achieve stable control.

**LQR Control**

LQR is an optimal control technique for linear dynamic systems that minimizes a cost function balancing state deviation and control effort. The notebook derives the equations of motion for the cartpole system, linearizes them, and applies LQR to compute optimal control inputs. The state is defined as ([x, \theta, \frac{dx}{dt}, \frac{d\theta}{dt}]), where (x) is the cart position, (\theta) is the pole angle, and (u) is the control input scaled by a gear ratio (k) to produce force (F = k \cdot u).
For a more advanced LQR application, see the Google DeepMind LQR tutorial.

**Cartpole Inverted Pendulum**

The cartpole system consists of:

- A cart of mass (M) sliding on a frictionless surface
- A massless pole of length (l) attached to the cart via a frictionless rotational joint
- A point mass (m) at the pole's tip
- A motor applying a force (F) in the (x)-direction to the cart

This setup is used to demonstrate both PID and LQR control strategies.

**Model Misspecification**

Real-world systems often deviate from idealized models due to:

- Linearization of nonlinear dynamics
- Inaccurate parameters (e.g., pole length)
- Time-varying properties (e.g., damping)

These errors, as noted by George Box ("All models are wrong, but some are useful"), can degrade controller performance. The notebook highlights the importance of robust control design in the presence of such uncertainties.
## Notebook Contents
The Jupyter notebook (``E03_Intro_to_Control1.ipynb``) includes:

1. **Package Installation:** Installs required Python libraries (``control``, ``mujoco``, ``mediapy``).
2. **Imports:** Sets up necessary libraries (``numpy``, ``control``,`` matplotlib``, ``mujoco``, ``mediapy``) and configures MuJoCo's rendering backend.
3. **MuJoCo Introduction:** Defines a simple MuJoCo simulation with a pendulum, box, and ball in XML format, demonstrating model creation, data handling, and rendering.
4. **Control Implementation:** Applies PID and LQR control to the cartpole system (specific details are in the notebook).
5. **Visualization:** Uses ``matplotlib`` and ``mediapy`` for plotting and rendering simulations.

## Prerequisites
To run the notebook, ensure you have the following:

- Python 3.11 or compatible version
- Jupyter Notebook or JupyterLab
- Required packages (installed via ``pip install control mujoco mediapy``)

## Usage

1. Clone this repository:
````
git clone <repository-url>
````

2. Navigate to the repository directory:
````
cd <repository-directory>
````

Launch Jupyter Notebook:
````
jupyter notebook
````

4. Open ``E03_Intro_to_Control1.ipynb`` and run the cells sequentially.
5. Experiment with PID and LQR parameters to observe their effects on the cartpole system.

## Notes

- The notebook assumes a Linux-based environment with ``egl`` as the MuJoCo rendering backend. For macOS, switch to ``glfw`` by modifying the environment variable (``%env MUJOCO_GL=glfw``).
- Fine-tuning PID and LQR parameters requires understanding the system dynamics and iterative testing.
- Model misspecification effects can be explored by intentionally altering parameters like pole length or mass.

## Resources

- MuJoCo Documentation
- Google DeepMind MuJoCo Tutorial
- Google DeepMind LQR Tutorial
