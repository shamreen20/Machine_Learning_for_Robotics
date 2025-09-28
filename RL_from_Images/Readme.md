# RL from Images using VAEs and DQN

This Jupyter notebook (`RL_from_Images.ipynb`) demonstrates the use of Variational Autoencoders (VAEs) for learning compact latent representations from image-based observations, combined with Deep Q-Networks (DQN) for reinforcement learning. It starts with training a VAE on the MNIST dataset for illustration, then applies the approach to RL in a pixel-observation version of the CartPole environment from Gymnasium.

The notebook covers:
- VAE concepts, architecture, and training.
- Integration of VAEs with DQN for handling image inputs in RL.
- Visualization of latent spaces and episode rollouts.

# Variational Autoencoders (VAEs)

Variational Autoencoders (VAEs) are a powerful class of generative models that extend the concept of traditional autoencoders, such as denoising autoencoders, by introducing a probabilistic approach to encoding and decoding data. While denoising autoencoders learn to reconstruct clean data from corrupted inputs, VAEs are designed to learn a meaningful, continuous, and structured latent representation of the data, enabling both reconstruction and generation of new samples.

## Key Concepts

- **Probabilistic Latent Space:**  
    Unlike standard autoencoders that map inputs to fixed points in the latent space, VAEs map inputs to a distribution (typically Gaussian) in the latent space. Each input is encoded as a mean and variance, allowing for sampling and generating diverse outputs.

- **Encoder and Decoder:**  
    The encoder network learns to approximate the posterior distribution of the latent variables given the input data. The decoder network reconstructs the input from samples drawn from the latent distribution.

- **Reparameterization Trick:**  
    To enable backpropagation through the stochastic sampling process, VAEs use the reparameterization trick, expressing the sampling operation as a deterministic function of the mean, variance, and a random noise variable.

- **Loss Function:**  
    The VAE loss combines two terms:  
    1. **Reconstruction Loss:** Measures how well the decoder reconstructs the input from the latent code (often using binary cross-entropy or mean squared error).  
    2. **KL Divergence:** Regularizes the learned latent distribution to be close to a standard normal distribution, encouraging smoothness and continuity in the latent space.

## Why Use VAEs?

- **Generative Modeling:**  
    VAEs can generate new, realistic samples by sampling from the latent space and decoding them, making them useful for tasks like image synthesis and data augmentation.

- **Structured Latent Space:**  
    The regularization encourages the latent space to be continuous and interpretable, allowing for smooth interpolation between data points and meaningful manipulations.

- **Uncertainty Quantification:**  
    By modeling distributions rather than points, VAEs can capture uncertainty in the data and generate diverse outputs.

## RL from Images

Reinforcement learning from images is a subfield of machine learning that focuses on training agents to make decisions based on visual input. In this approach, the agent directly learns a policy or value function from raw image data, without the need for handcrafted features or explicit state representations. This allows the agent to perceive and understand the environment solely through visual observations.

When it comes to training a reinforcement learning agent from images, there are two main options: using an end-to-end learning algorithm or separating perception and control.

In end-to-end learning, the agent learns to directly map raw image observations to actions. This approach has the advantage of simplicity and can potentially capture complex relationships between images and actions. However, it often requires a large amount of training data and can be computationally expensive.

On the other hand, separating perception and control involves training separate models for perception (e.g., image recognition) and control (e.g., decision-making). The perception model processes the raw images and extracts relevant features, which are then used as input to the control model. This approach can be more interpretable and efficient, as it allows for leveraging existing image recognition techniques and reduces the dimensionality of the problem. However, it requires careful design and engineering of the perception and control modules.

In summary, the choice between end-to-end learning and separating perception and control depends on the specific requirements of the task and the available resources. End-to-end learning offers simplicity but requires more data and computational resources, while separating perception and control provides interpretability and efficiency but requires additional design and engineering efforts.

The notebook specifically integrates VAEs for perception (latent state extraction) with DQN for control, applied to pixel-based environments like CartPole and conceptually extendable to Atari games.

## Requirements

The notebook requires Python 3.11+ and the following packages (installed via pip in the first cell):

- `swig`
- `mediapy`
- `tqdm`
- `mujoco==3.1.4`
- `torch`
- `torchrl`
- `gymnasium[box2d,mujoco]==0.28.1`
- Additional dependencies (implicit via imports): `numpy`, `matplotlib`, `torchvision`

To set up the environment:
```
pip install -r requirements.txt
```
(You can generate `requirements.txt` from the notebook's install commands.)

GPU acceleration is recommended for training (uses CUDA if available).

## Usage

1. Open the notebook in Jupyter Lab or Jupyter Notebook:
   ```
   jupyter lab RL_from_Images.ipynb
   ```

2. Run cells sequentially:
   - Install packages (first cell).
   - Import libraries and define helper functions.
   - Train the VAE on MNIST (takes ~10-15 minutes on CPU; faster on GPU).
   - Set up the CartPole environment with pixel observations.
   - Train the VAE-DQN agent (configurable episodes and steps).
   - Render and visualize episodes, including latent space traversals.

Key hyperparameters:
- VAE: Latent dimension (`latent_dim=2` for visualization), epochs=100.
- DQN: Replay buffer size=100,000; batch size=128; gamma=0.99; epsilon decay.
- Environment: CartPole-v1 with pixel observations (84x84 grayscale).

## Results

- **VAE on MNIST**: Achieves low reconstruction loss (~149 after 100 epochs). Visualizes latent space clustering by digit classes.
- **RL Training**: Trains a VAE-DQN agent on pixel-based CartPole, achieving stable rewards (visualized in plots).
- **Visualizations**:
  - Episode rollouts as videos.
  - Side-by-side comparisons of original frames, VAE reconstructions, and latent states (2D scatter or bar plots).
  
Example output (from notebook):
- Total reward per episode during training.
- Interactive video of agent performance.

## Limitations and Extensions

- The VAE is pre-trained on MNIST for demo purposes; in practice, train on environment frames.
- Latent dim=2 aids visualization but may limit performance; try higher dims (e.g., 32).
- Extend to more complex environments (e.g., Atari games) or advanced RL (e.g., DDQN, Rainbow).
- Potential improvements: Use convolutional layers in DQN, or integrate with actor-critic methods.

## References

Citations are rendered inline above. For full details, refer to the linked sources from the searches.