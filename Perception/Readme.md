# E10_Perception: Robotics Perception with Kalman Filters and Denoising Autoencoders

## Overview

This Jupyter notebook (``E10_Perception.ipynb``) explores perception in robotics, focusing on handling noisy sensor data and partial observations. It implements two approaches to mitigate noise:

1. **Kalman Filter:** A model-based algorithm for state estimation from noisy measurements, applied to 1D and 2D localization scenarios.
2. **Denoising Autoencoder:** A neural network-based method to learn noise removal from high-dimensional data (MNIST images) without explicit dynamic models.

The notebook includes visualizations (e.g., uncertainty propagation, latent spaces) and tasks to deepen understanding of these techniques. Topics covered include sensor noise, partial observation, state estimation, latent spaces, and generative modeling.

## Objectives

- Demonstrate how Kalman Filters estimate robot states (position, velocity) from noisy measurements.
- Show how Denoising Autoencoders learn to remove noise from images, capturing data structure in a latent space.
- Analyze the trade-offs between model-based (Kalman Filter) and data-driven (Autoencoder) approaches.
- Explore generative capabilities of autoencoders and the impact of latent space dimensionality on reconstruction quality.

## Prerequisites
- **Python:** Version 3.10 or compatible.
- **Hardware:** GPU recommended for faster autoencoder training (CUDA-compatible).
- **Dependencies:** Install required packages using the provided commands.

## Installation

To set up the environment, run the following commands in a terminal or within the notebook:
````
pip install swig mediapy
pip install mujoco==3.1.4
pip install torch torchrl
pip install gymnasium[box2d,mujoco]==0.28.1
````
These install:
    - ``swig``, ``mediapy``: For rendering and visualization.
    - ``mujoco``: Physics simulation (not used in this notebook but included).
    - ``torch``, ``torchrl``: PyTorch for neural networks and reinforcement learning utilities.
    - ``gymnasium``: Environment for reinforcement learning (used in render_episode).

Additional dependencies (included in the notebook):

``numpy``, ``matplotlib``: For numerical operations and plotting.
``torchvision``: For MNIST dataset and transforms.

Usage
1. **Open the Notebook:**
- Use Jupyter Notebook or JupyterLab:
````
jupyter notebook E10_Perception.ipynb
````
- Alternatively, open in Google Colab or VS Code with Jupyter support.

2. **Run Cells Sequentially:
**
    - Execute the installation cell first if dependencies are not installed.
    - Run import and function definition cells.
    - Execute Kalman Filter and Denoising Autoencoder sections to see results.
    - Complete the tasks to explore parameters and latent spaces.

3. **View Outputs:**

- Visualizations include:

    - Kalman Filter: Animated uncertainty propagation for 1D and 2D trajectories.
    - Autoencoder: Original, noisy, and denoised MNIST images; 2D latent space scatter plot.

- Console outputs show training loss for the autoencoder.

## Topics Discussed

- **Perception in Robotics:** Sensing and interpreting environments using noisy sensors.
- **Noisy Sensor Data:** Causes (sensor limitations, interference) and mitigation (filtering, fusion).
- **Partial Observation:** Challenges of incomplete data and solutions (probabilistic modeling, machine learning).
- **Kalman Filter:** State estimation, sensor fusion, parameter tuning, handling missing data.
- **Denoising Autoencoders:** Learning-based noise removal, latent space representation, gerative modeling.
- **Latent Spaces:** Dimensionality reduction, feature extraction, applications in reinforcement learning.
- **Generative Models:** Creating new data samples from latent spaces.
- **Neural Network Design:** Trade-offs between compression (latent dimension) and performance (reconstruction quality).

## Acknowledgments
- Inspired by robotics perception challenges and deep learning techniques.
- Built with PyTorch, NumPy, Matplotlib, and Torchvision.
- Thanks to the open-source community for providing robust libraries.