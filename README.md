# Data Intelligence Challenge – 2AMC15 Assignment 2

This repository (https://github.com/AdminMas7er/2AMC15_Assignment2_2025) contains the environment and training code for autonomous food delivery agents in a continuous-space restaurant settings. It supports two RL agents, Deep Q-Learning (DQN) and Soft Actor-Critic (SAC), and provides tools for restaurant creation, training, and evaluation.

---

## Quickstart Guide

### 1. Set up environment
We recommend a conda Python ≥ 3.10 installation (tested with 3.11):

```bash
conda create -n dic2025_2 python=3.11
conda activate dic2025_2
pip install -r requirements.txt
````

### 2. Run a baseline (grid world)

```bash
python train.py --agent random_agent --restaurant grid_configs/my_first_restaurant.npz
```

---

## Continuous Environment Setup

### 1. Create a restaurant layout

To start the interactive table placement tool:

```bash
python world/restaurant_creator_localhost.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser to create a custom restaurant with tables and a kitchen and export your layout as `.npz`.

Absolutely! Below is a **more expansive and structured** version of the README starting from **“2. Train a DQN agent”**. It now includes all options, better guidance, and clear links between code files and functionality.

You can paste this directly into your `README.md`.

---

### 2. Train a DQN agent

To train a Deep Q-Learning (DQN) agent in a continuous-space restaurant environment, use the following script:

```bash
python train_dqn.py --restaurant grid_configs/my_first_restaurant.npz
````

This script trains the agent using a minimal observation space and 5 discrete actions. Results and models will be saved in a timestamped folder under `results/`.

#### Full options for `train_dqn.py`

```bash
usage: train_dqn.py --restaurant RESTAURANT [--episodes EPISODES]
                    [--no_gui] [--seed SEED]
                    [--load_model PATH] [--demo]
```

* `--restaurant`: Path to the `.npz` file generated via the restaurant creator GUI.
* `--episodes`: Number of training episodes (default: 100).
* `--no_gui`: Disable the visual GUI window during training (useful for faster headless training).
* `--seed`: Set the random seed for reproducibility.
* `--load_model`: Load a previously trained model (used to resume training or run in demo mode).
* `--demo`: Run the loaded model in inference mode (no learning or exploration).

#### Example usage:

```bash
# Train a new DQN agent without GUI
python train_dqn.py --restaurant restaurants/example_layout.npz --episodes 500 --no_gui

# Resume training from checkpoint
python train_dqn.py --restaurant restaurants/example_layout.npz --load_model results/dqn_training_TIMESTAMP/models/best_model.pth

# Run trained model in demo mode
python train_dqn.py --restaurant restaurants/example_layout.npz --load_model results/.../best_model.pth --demo
```

---

### 3. Train a SAC agent

To train a Soft Actor-Critic (SAC) agent in the same environment:

```bash
python train_sac.py --restaurant path/to/layout.npz --episodes 200
```

The SAC agent supports entropy-regularized learning and is adapted to a discrete action space. Like `train_dqn.py`, the training process logs performance metrics and saves models under the `results/` directory.

#### Full options for `train_sac.py`

```bash
usage: train_sac.py [--restaurant RESTAURANT] [--episodes EPISODES] 
                    [--max_steps MAX_STEPS] [--eval_freq EVAL_FREQ]
                    [--eval_episodes EVAL_EPISODES] [--no_gui]
                    [--render_eval] [--seed SEED]
                    [--output_dir OUTPUT_DIR] [--load_model MODEL]

SAC Hyperparameters:
  --lr LR
  --gamma GAMMA
  --tau TAU
  --buffer_size BUFFER_SIZE
  --batch_size BATCH_SIZE
  --hidden_size HIDDEN_SIZE
```

* `--restaurant`: Path to the `.npz` layout file (default: grid\_configs/my\_first\_restaurant.npz)
* `--episodes`: Total training episodes (default: 2000)
* `--max_steps`: Maximum steps per episode (default: 500)
* `--eval_freq`: Evaluation frequency (default: every 100 episodes)
* `--eval_episodes`: How many episodes to run during each evaluation phase
* `--no_gui`: Disable GUI during training
* `--render_eval`: Enable rendering during evaluation only
* `--seed`: Set random seed for reproducibility
* `--output_dir`: Output folder for results (default: `results/`)
* `--load_model`: Path to a saved model to resume training or run demo

#### SAC hyperparameter overrides

* `--lr`: Learning rate (default: 1e-3)
* `--gamma`: Discount factor (default: 0.98)
* `--tau`: Target network soft update coefficient (default: 0.01)
* `--buffer_size`: Replay buffer size (default: 50,000)
* `--batch_size`: Batch size for training (default: 128)
* `--hidden_size`: Hidden layer size for actor/critic networks (default: 512)

#### Example usage:

```bash
# Train a SAC agent with default settings
python train_sac.py --restaurant grid_configs/my_first_restaurant.npz --episodes 1000

# Train with rendering disabled for speed
python train_sac.py --restaurant grid_configs/my_first_restaurant.npz --episodes 1000 --no_gui

# Test a trained SAC model
python train_sac.py --load_model results/sac_training_TIMESTAMP/models/best_model.pth --episodes 10 --demo
```

---
<!-- 
## Results and Logging

Both training scripts will automatically create output folders under:

```
results/sac_training_TIMESTAMP/
results/dqn_training_TIMESTAMP/
```

These folders contain:

* `models/`: Saved model checkpoints
* `log.csv`: Per-episode training logs (reward, success, steps, etc.)
* `plots/`: Auto-generated performance plots (reward, success rate, episode length)
* `config.json`: Metadata and hyperparameters used for the run

--- -->

## How the Code Works (File Guide)

This section gives a minimal overview of how to work with the codebase.

### Environments

* `world/environment_stefan.py`
  Core environment class (`ContinuousEnvironment`) for continuous-space delivery tasks. Handles robot movement, collision detection, table layouts, and rendering.

* `world/continuous_space.py`
  Defines the restaurant's physical space (width, height, table positions, etc.). Used to create `.npz` layout files.

* `world/restaurant_creator_localhost.py`
  Launches a local web app to visually place tables. Exported layouts can be used for training.

### Agents

* `agents/sac_agent.py`
  Discrete-action SAC agent with automatic entropy tuning. Includes actor/critic networks, replay buffer, and training logic.

* `agents/dqn_final.py`
  DQN agent implementation using target networks and \$\epsilon\$-greedy exploration.

* `agents/random_agent.py`
  Baseline agent that takes random actions.

### Training Scripts

* `train_sac.py`
  Trains a SAC agent in a specified continuous environment.

* `train_dqn.py`
  Trains a DQN agent using the same environment and interfaces.

* `train.py`
  Original training script for simple agents (not used for RL training).

---