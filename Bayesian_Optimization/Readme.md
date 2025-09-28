# Bayesian Optimization Notebook

This Jupyter notebook (`BayesianOptimization.ipynb`) provides a practical implementation of Bayesian optimization for tuning a PID controller in the InvertedPendulum-v4 environment from Gymnasium's MuJoCo suite using Meta's Ax library. It demonstrates multi-objective optimization to maximize cumulative return (reward) while minimizing cumulative energy usage, using Gaussian processes as surrogates and acquisition functions for efficient search. The notebook applies Bayesian optimization to a control problem, balancing performance and efficiency in a simulated physics environment.

The notebook includes code for package installation, controller definition, evaluation, Ax setup, optimization loop, and rendering of the optimized policy's performance.

## Key Components

### 1. Environment and Dependencies

The notebook uses the InvertedPendulum-v4 environment, a MuJoCo task where a cart balances an inverted pendulum upright. Dependencies are installed via pip, including:
- Swig for building extensions.
- Mediapy for video rendering.
- Ax-platform (v0.5.0 in notebook; latest is 1.1.2 as of September 2025) for Bayesian optimization.
- Gymnasium (with MuJoCo extras) for the environment.
- Implicit dependencies: NumPy, SciPy, Pandas, Matplotlib, SymPy, PyTorch, BoTorch, GPyTorch.

### 2. PID Controller and Evaluation

A PID (Proportional-Integral-Derivative) controller is defined for cart position (`x`) and pendulum angle (`theta`). The evaluation function runs episodes in the environment, computing average cumulative reward and energy (based on action magnitudes).

Key functions:
- `controller(s, parameters)`: Computes action from state and PID params.
- `evaluate_pi(parameters)`: Simulates episodes, returns cum_return and cum_energy.

### 3. Optimization with Ax

The core uses Ax for multi-objective Bayesian optimization. It sets up an experiment with parameters (`p_x`, `d_x`, `p_theta`, `d_theta`) and objectives (maximize return with threshold 500, minimize energy with threshold 1000). Runs 40 trials, selects Pareto-optimal parameters.

Example configuration (excerpt from notebook):
```python
ax_client.create_experiment(
    name='cartpole',
    parameters=[
        {'name': 'p_x', 'type': 'range', 'bounds': [-5., 5.]},
        {'name': 'd_x', 'type': 'range', 'bounds': [-10., 5.]},
        {'name': 'p_theta', 'type': 'range', 'bounds': [-40., 0.]},
        {'name': 'd_theta', 'type': 'range', 'bounds': [-10., 5.]},
    ],
    objectives={"cum_return": ObjectiveProperties(minimize=False, threshold=500.0),
                "cum_energy": ObjectiveProperties(minimize=True, threshold=1000.0)},
)
```

### 4. Rendering and Visualization

After optimization, loads the best parameters and renders an episode:
```python
best_params = list(ax_client.get_pareto_optimal_parameters().values())[0][0]
render_episode(lambda s: controller(s, best_params), "InvertedPendulum-v4")
```
This embeds an HTML video of the balanced pendulum using Mediapy.

### 5. Tasks
- **Main Task**: Tune PID parameters for InvertedPendulum-v4 using Ax for multi-objective optimization.
- **Extension (Homework)**: Apply to other MuJoCo environments like CartPole-v1 or HalfCheetah-v4. Refer to [Ax documentation](https://ax.dev/) and [Gymnasium MuJoCo](https://gymnasium.farama.org/environments/mujoco/).

## Requirements

To run this notebook, ensure Python 3.11+ and install dependencies:
```bash
pip install swig mediapy ax-platform==0.5.0 gymnasium[mujoco]
```
Note: For the latest Ax (v1.1.2), use `pip install ax-platform`. The notebook assumes a compatible environment; MuJoCo may require additional setup for rendering. Runs on CPU or GPU.

## Usage

1. Download the notebook.
2. Install dependencies as above.
3. Run in Jupyter Notebook, JupyterLab, or Google Colab.
4. Execute cells sequentially: install, define functions, optimize, render.
5. Experiment by adjusting trials, parameters, or objectives.

## Results

The optimized PID controller balances the pendulum effectively, achieving high return with low energy. The notebook prints the best parameters (e.g., {'p_x': ..., 'd_x': ..., ...}) and displays a video of the episode. Optimization converges to Pareto-optimal solutions after 40 trials.

For more on Bayesian optimization:
- Gaussian Processes: Flexible surrogates for modeling (see [GP book](https://gaussianprocess.org/gpml/)).
- Acquisition Functions: EI, UCB for exploration-exploitation.
- Ax/BoTorch: See [BoTorch paper](https://arxiv.org/abs/1910.06403).

If updating Ax, check API changes in the docs.

## License

This notebook is for educational purposes. Dependencies follow their licenses (e.g., Ax under MIT, Gymnasium under MIT).