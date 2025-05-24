# Machine Learning Model Optimization

This repository contains a Jupyter notebook demonstrating hyperparameter optimization with Optuna, Convolutional Neural Networks (CNNs) for CIFAR-10 classification, and Long Short-Term Memory (LSTM) networks for Apple stock price prediction.

## Overview

- **Hyperparameter Optimization:** Uses Optuna for Bayesian optimization to tune hyperparameters, including a 2D quadratic function minimization and CNN hyperparameter tuning.
- **Convolutional Neural Networks (CNNs):** Implements a CNN for CIFAR-10 classification, enhanced with Dropout, BatchNorm, Residual Connections, and data augmentations to achieve ~90% accuracy.
- **Recurrent Neural Networks (RNNs):** Uses an LSTM to predict Apple stock prices, with experiments on architecture and additional features like trading volume.

## Requirements

- Python 3.8+
- PyTorch (torch, torchvision)
- Optuna
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Installation

1. Clone the repository:
````
git clone https://github.com/your-repo/machine-learning-optimization.git
cd machine-learning-optimization
````

2. Install dependencies:
````
pip install torch torchvision optuna pandas numpy scikit-learn matplotlib
````

## Usage

1. Open the Notebook:

- Launch Jupyter Notebook:
````
jupyter notebook ML_Optimization.ipynb
````

- The notebook contains three main sections: Optuna optimization, CNN for CIFAR-10, and LSTM for stock prediction.

2. Run the Code:

- Optuna Section: Minimize a 2D quadratic function and tune CNN hyperparameters (learning rate, batch size, dropout). Reduce n_trials or epochs for faster execution.
- CNN Section: Train a CNN on CIFAR-10 with augmentations, BatchNorm, Dropout, and Residual Connections. Visualize filters and evaluate accuracy (~85-90%).
- LSTM Section: Predict Apple stock prices using an LSTM. Experiment with layers, hidden size, or additional features like volume.

3. Data:

- CIFAR-10 is automatically downloaded via torchvision.
- For LSTM, provide a CSV file (AAPL.csv) with columns: Date, Close, Volume (e.g., from Yahoo Finance).

## Tasks and Findings
**Hyperparameter Optimization**

- **Contour Plot:** Optuna efficiently samples the 2D quadratic function space, converging near the minimum (0,0).
- **Trial Pruning:** Unpromising trials are terminated early to save compute, based on intermediate metrics.
- **Epochs:** Best CNN configurations use ~20 epochs per trial to balance training and efficiency.
- **Challenges:** Tuning large models (e.g., LLMs) is compute-intensive with complex hyperparameter spaces. Solutions include pruning and transfer learning.


**CNNs (CIFAR-10)**

- **Parameters:** CNN (36,552–200,000 parameters) vs. Feed-Forward (3.8M). CNNs are efficient due to parameter sharing and local connectivity.
- **90% Accuracy:** Achieved with deeper CNNs, Dropout, BatchNorm, Residual Connections, L2 regularization, and augmentations. Optuna tunes hyperparameters.
- **Test Accuracy:** Expect ~87-90% on a separate test set, slightly lower than validation due to tuning bias.
- **Conv1 Output:** Learns edges, textures, and color patterns, visible in feature map visualizations.

**RNNs (Stock Prediction)**

- **Dataset Split:** Sequential 80/20 train/test split preserves temporal order, unlike random splits for CNNs.
- **LSTM Modifications:** Adding layers or hidden units slightly improves accuracy but risks overfitting.
- **Additional Features:** Including volume or correlated stock prices reduces MSE by ~10-20%.

## License
This notebook is provided for educational purposes. Ensure compliance with the licensing terms of PyTorch, NumPy, Matplotlib, and torchvision.

