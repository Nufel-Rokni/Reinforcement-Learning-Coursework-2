# Reinforcement-Learning-Coursework-2

# A2C Atari Agent – MsPacman

This repository contains a simple implementation of an -- Advantage Actor-Critic (A2C) -- agent for the Atari game Ms. Pacman, using `gymnasium`, `ale-py`, and `PyTorch`. The code includes an environment factory, a convolutional neural network with shared policy/value heads, and a training loop with debugging outputs.

## Environment

This code was developed and tested in -- Kaggle Colab -- (Kaggle's hosted Jupyter notebook environment, similar to Google Colab).  
**Important:**  
Because of this, the dependencies and library versions are aligned with the Kaggle/Colab environment. Running this code outside of Kaggle/Colab (e.g., on a local machine or a different cloud setup) [might result in version mismatches, missing packages, or crashes.]

Key dependencies:
- `gymnasium`
- `ale-py`
- `torch`
- `numpy`

## How to Run

To run the code:
1. Copy the entire Python script into a Kaggle Notebook or Colab Notebook.
2. Make sure to install the required packages (most are pre-installed in Kaggle):
   ```bash
   pip install gymnasium ale-py torch numpy

## Training and Performance Analysis

### Reward Progression

The figure below illustrates the reward per episode (gold) alongside a 100-episode moving average (red). Initially, the moving average quickly rises from about 200 to roughly 550 within 200 episodes, after which it stabilizes between 500 and 600, suggesting consistent average performance. The significant variance in individual episode rewards, with occasional peaks around 2500 and drops near 0, reflects the inherent randomness of MsPacman's environment.

![Training Reward Over Episodes](Training%20reward%20episodes.png)

### Reward Distribution

This figure compares the reward distributions from early training (episodes 1–500, gold) and late training (episodes 2915–3414, orange). Initially, most rewards cluster between 200 and 400, whereas later in training, the distribution shifts significantly higher, frequently exceeding 600 points with some episodes surpassing 1500. This progression highlights the agent's improved skill over time.

![Reward Distribution Early vs Late](Reward%20distribution.png)

### Losses and Entropy Dynamics

The following plot displays policy loss (yellow), value loss (orange), entropy (red), and total loss (pink) over the first 100 training blocks (approximately 10 episodes each). Initially, all losses, especially value loss, spike notably, then decline significantly by block 20. Entropy gradually decreases from around 2.5 to about 1.0. Recurring loss spikes correspond to infrequent but significant reward episodes.

![Losses and Entropy](losses%20and%20entropy%20over%20training.png)

### Loss vs. Reward Correlation

Below is a scatter plot of total loss versus the 100-episode moving average reward. Although lower losses generally align with higher average rewards, the broad scatter indicates that loss isn't strictly predictive of performance, as occasional high-loss events coincide with good performance metrics.

![Total Loss vs Moving Average Reward](total%20loss%20vs%20average%20moving%20reward.png)

### Smoothed Loss Trend

The rolling 5-episode average of total loss across training episodes is shown below. Initially dropping significantly by episode 500, loss continues fluctuating moderately (between 5 and 25) throughout later episodes, reflecting the agent's continuous adaptation to varying game conditions.

![Rolling Average of Total Loss](Rolling%20average%20of%20total%20loss.png)

