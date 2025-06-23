# Machine Learning for Robotics Tutorial 1
## Overview
This Jupyter Notebook, Machine_learning_for_Robotics_tutorial_1.ipynb, is an introductory tutorial on using PyTorch for machine learning in robotics applications. It covers the fundamentals of PyTorch tensors, the core data structure for building and training neural networks, and progresses to advanced topics like neural network construction and dataset handling. The notebook includes tensor instantiation, operations, broadcasting, GPU/CPU computation, Autograd for gradient computation, nn.Module for neural network architectures, and the use of MNIST and CIFAR-10 datasets for training models. 

## Prerequisites
To run this notebook, ensure you have the following:

- **Python 3.x:** The notebook is designed for Python 3.
- **Jupyter Notebook:** Required to open and run the .ipynb file.
- **PyTorch Libraries:**
  - **torch:** Core library for tensor operations and Autograd (automatic 
    differentiation for gradient computation).
  - **torch.nn:** Provides neural network modules, including nn.Module for defining 
    custom neural network architectures.
  - **torch.nn.functional:** Contains functions like activation functions (e.g., 
    F.relu) and loss functions for neural networks.
  - **torch.linalg:** Offers linear algebra operations (e.g., Cholesky 
    decomposition).
  - **torchvision:** Utilities for computer vision tasks, including datasets like 
  - **MNIST** and **CIFAR-10**, and image transformations.


- **NumPy:** For numerical computations.
- **Matplotlib:** For plotting visualizations.
- **GPU (Optional):** A CUDA-enabled GPU is recommended for GPU-related tasks and 
  neural network training, but the code falls back to CPU if a GPU is 
  unavailable.

## Datasets
The notebook uses the following datasets from torchvision.datasets for training neural networks, particularly in later sections:

- **MNIST:** A dataset of 28x28 grayscale images of handwritten digits (0-9), 
  used for image classification tasks in robotics, such as recognizing numerical 
  inputs.

- **CIFAR-10:** A dataset of 32x32 color images across 10 classes (e.g., 
  airplane, cat, dog), used for more complex computer vision tasks like object 
  recognition.Additionally, synthetic data (e.g., a 2D grid of points X) is 
  generated for early tasks, such as plotting a linear function.


## PyTorch Features

- **Autograd:** PyTorch’s automatic differentiation engine (torch) computes gradients 
  for tensor operations, enabling optimization techniques like gradient descent 
  for training neural networks.
- **nn.Module:** A base class in torch.nn used to define neural network 
  architectures, encapsulating layers, parameters, and forward passes for models 
  applied to datasets like MNIST or CIFAR-10.
- **Neural Networks:** The notebook builds neural networks using torch.nn and 
  nn.Module, leveraging torch.nn.functional for activation functions and loss - 
  computations, applied to tasks like image classification with MNIST and CIFAR- 10.
- **Tensor Operations:** Core tensor operations (e.g., element-wise addition, torch.matmul) 
  are demonstrated with torch and torch.linalg, forming the foundation for neural network 
  computations.

## Installation

1. Install Python: Ensure Python 3.x is installed. Download from python.org if needed.
2. Install Jupyter: Run the following command to install Jupyter Notebook:
````
pip install jupyter
````

3. Install Dependencies: Install the required Python packages:
````
pip install torch numpy matplotlib torchvision
````

4. Set Up GPU (Optional): For GPU support, ensure a CUDA-enabled GPU is available and 
   install PyTorch with CUDA support. Refer to PyTorch's official website for the correct 
   installation command.

## Usage

1. Open the Notebook:

- Locally, navigate to the notebook directory and run:
````
jupyter notebook Machine_learning_for_Robotics_tutorial_1.ipynb
````

- In Google Colab, upload the notebook and select a GPU runtime (Runtime > Change runtime 
  type > GPU).


2. Run the Cells:

- Execute the cells sequentially to follow the tutorial.
- The notebook includes:
  - **Imports:** Libraries including ``torch``, ``torch.nn``, ``nn.Module``, ``torch.linalg``, 
    ``torchvision``, ``numpy``, ``matplotlib.pyplot``, ``time``, and ``os``.
  - **Plotting Function:** ``plot_linear_regression`` for visualizing linear regression results, 
    adaptable for neural network training visualizations.
  - **Tensor Introduction:** Demonstrates tensor creation (e.g., torch.tensor), operations 
    (e.g., element-wise addition, torch.matmul), broadcasting, and GPU/CPU handling.
  - N**eural Network Development:** Uses ``nn.Module`` and ``torch.nn.functional`` to build and train 
    neural networks, with Autograd for optimization, applied to MNIST and CIFAR-10 
    datasets.


## Key Learning Objectives

- Master PyTorch tensors and operations using ``torch`` and ``torch.linalg``.
- Build and train neural networks with ``torch.nn``, ``nn.Module``, and ``torch.nn.functional``.
- Use Autograd for gradient-based optimization in neural network training.
- Work with computer vision datasets (MNIST, CIFAR-10) via ``torchvision`` for robotics 
  applications.
- Handle broadcasting and avoid related errors.
- Manage GPU and CPU tensors for device compatibility.
- Visualize data and model outputs using Matplotlib (e.g., 3D scatter plots, loss curves).

## Notes

- The notebook assumes basic knowledge of Python, linear algebra, and machine learning 
  concepts.
- GPU tasks require a CUDA-compatible GPU; otherwise, the code defaults to CPU.
- The plot_linear_regression function is designed for linear regression visualization but 
  can be adapted for neural network training results.
- MNIST and CIFAR-10 datasets require internet access for downloading via 
  ``torchvision.datasets``.
- Refer to the PyTorch documentation for details on ``torch``, ``torch.nn``, ``torch.nn.functional``, ``torch.linalg``, and ``torchvision``.

## Troubleshooting

- **GPU Error:** If you see a "tensors not on the same device" error, ensure all tensors are 
  moved to the same device using ``.to("cuda")`` or ``.to("cpu")``.
- **Module Not Found:** Verify that ``torch``, ``numpy``, ``matplotlib``, and ``torchvision ``are installed.
- **Colab GPU Setup:** In Colab, enable GPU via Runtime > Change runtime type > GPU.
- **Dataset Access:** Ensure internet access for downloading MNIST/CIFAR-10 datasets via ``torchvision.datasets``.

## License
This notebook is provided for educational purposes. Ensure compliance with the licensing terms of PyTorch, NumPy, Matplotlib, and torchvision.
