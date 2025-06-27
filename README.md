# Welcome to Data Intelligence Challenge-2AMC15 Assignment 2!

This is the repository containing the challenge environment code for both grid-world and continuous-space navigation scenarios.

---

## Quickstart

1. Create a virtual environment with Python ≥ 3.10 (we recommend 3.11):
   `conda create -n dic2025_2 python=3.11`
   Then activate it:
   `conda activate dic2025_2`
2. Clone the repository (or download it):
   `git clone https://github.com/AdminMas7er/2AMC15_Assignment2_2025.git`
3. Install dependencies:
   `pip install -r requirements.txt`
4. Run the grid-world training script:
   `python train.py --agent random_agent --restaurant grid_configs/my_first_restaurant.npz`

---

## Training in Continuous Environments

You can also now train agents in continuous 2D restaurant environments using the new continuous space simulation:

1. First, create a restaurant space interactively:
   `python world/restaurant_creator_localhost.py`
   Then go to [127.0.0.1:5000](http://127.0.0.1:5000) or http://localhost:5000 to place tables and export your layout as `.npz`.

2. Run training or testing on the saved `.npz` space:

   ```bash
   python train.py --agent random_agent --restaurant grid_configs/my_first_restaurant.npz --iter 500
   ```

### train_dqn.py usage for continuous space:

```bash
usage: train_dqn.py --restaurant RESTAURANT [--episodes EPISODES]
                    [--no_gui] [--seed SEED]
                    [--load_model PATH] [--demo]

Train agent in a continuous restaurant space.

options:
  --restaurant RESTAURANT Path to the .npz restaurant layout
  --episodes EPISODES             Number of training episodes (default: 100)
  --no_gui                Disables GUI rendering
  --seed SEED             Random seed (default: 42)
  --load_model PATH         Path to a saved model to resume training or run demo
  --demo                    Run in demo mode (no exploration)
```

### train.py usage for continuous space:

```bash
usage: train.py --agent AGENT --restaurant RESTAURANT [--iter ITER]
                [--no_gui] [--seed SEED]

Train agent in a continuous restaurant space.

options:
  --agent AGENT           Name of the agent module (e.g., random_agent)
  --restaurant RESTAURANT Path to the .npz restaurant layout
  --iter ITER             Number of iterations (default: 1000)
  --no_gui                Disables GUI rendering
  --seed SEED             Random seed (default: 42)
```

---

## SAC (Soft Actor-Critic) Agent Training

This repository includes a high-performance SAC implementation specifically designed for the continuous restaurant delivery environment. SAC offers superior sample efficiency and stability compared to other algorithms.

### Quick Start with SAC

1. **Basic Training** (recommended settings):
   ```bash
   python train_sac.py --episodes 100 --max_steps 300 --eval_freq 25 --no_gui
   ```

2. **Training with GUI** (to visualize the learning process):
   ```bash
   python train_sac.py --episodes 100 --max_steps 300 --eval_freq 25
   ```

3. **Extended Training** (for better performance):
   ```bash
   python train_sac.py --episodes 200 --max_steps 500 --eval_freq 50 --no_gui
   ```

### SAC Training Options

```bash
usage: train_sac.py [--restaurant RESTAURANT] [--episodes EPISODES] 
                    [--max_steps MAX_STEPS] [--eval_freq EVAL_FREQ]
                    [--no_gui] [--output_dir OUTPUT_DIR] [--load_model MODEL]
                    [--lr LR] [--gamma GAMMA] [--batch_size BATCH_SIZE]

Train SAC agent in continuous restaurant environment

options:
  --restaurant RESTAURANT     Path to restaurant layout (default: grid_configs/my_first_restaurant.npz)
  --episodes EPISODES         Number of training episodes (default: 2000)
  --max_steps MAX_STEPS       Maximum steps per episode (default: 500)
  --eval_freq EVAL_FREQ       Evaluation frequency in episodes (default: 100)
  --eval_episodes EVAL_EPISODES  Episodes for evaluation (default: 10)
  --no_gui                    Disable GUI during training
  --render_eval               Enable rendering during evaluation
  --seed SEED                 Random seed for reproducibility (default: 42)
  --output_dir OUTPUT_DIR     Output directory for results (default: results)
  --load_model MODEL          Path to pre-trained model to continue training
  
SAC Hyperparameters:
  --lr LR                     Learning rate (default: 1e-3)
  --gamma GAMMA               Discount factor (default: 0.98)
  --tau TAU                   Soft update coefficient (default: 0.01)
  --buffer_size BUFFER_SIZE   Replay buffer size (default: 50000)
  --batch_size BATCH_SIZE     Training batch size (default: 128)
  --hidden_size HIDDEN_SIZE   Neural network hidden layer size (default: 512)
```

### Understanding SAC Results

After training, results are saved in a timestamped directory (e.g., `results/sac_training_20241227_143022/`):

**Directory Structure:**
```
results/sac_training_TIMESTAMP/
├── models/
│   ├── best_model.pth          # Best performing model during training
│   └── final_model.pth         # Final model at end of training
├── plots/
│   └── sac_training_progress.png  # Training curves and metrics
└── logs/
    ├── config.json             # Training configuration
    ├── training_log.json       # Detailed training data
    └── final_results.json      # Summary of final performance
```

**Key Performance Metrics:**
- **Success Rate**: Percentage of episodes where the agent successfully delivered orders
- **Mean Reward**: Average reward per episode (higher is better)
- **Mean Length**: Average number of steps per episode (lower indicates efficiency)
- **Temperature (α)**: SAC's exploration parameter (automatically tuned)

### Loading and Testing Trained Models

1. **Test a trained model**:
   ```bash
   python train_sac.py --load_model results/sac_training_TIMESTAMP/models/best_model.pth --episodes 50 --no_gui
   ```

2. **Visualize trained model performance**:
   ```bash
   python train_sac.py --load_model results/sac_training_TIMESTAMP/models/best_model.pth --episodes 10 --render_eval
   ```

### SAC Implementation Features

- **Enhanced State Representation**: 26-dimensional state vector including position, heading, target information, spatial awareness, and navigation features
- **Prioritized Experience Replay**: Successful experiences are given higher sampling priority for improved learning
- **Automatic Temperature Tuning**: Exploration-exploitation balance is automatically optimized
- **Gradient Clipping**: Ensures stable training by preventing gradient explosions
- **Learning Rate Scheduling**: Gradual learning rate decay for stable convergence

### Troubleshooting

**Common Issues:**

1. **Low Success Rate**: Try increasing `--episodes` or adjusting `--lr` (e.g., `--lr 3e-4`)
2. **Training Too Slow**: Use `--no_gui` and reduce `--eval_freq`
3. **Unstable Learning**: Decrease learning rate `--lr 1e-4` or increase `--batch_size 256`

**Performance Tips:**
- Use `--no_gui` for faster training
- Start with default hyperparameters before tuning
- Monitor the temperature (α) value - it should decrease during training
- Success rates typically improve after 50-75 episodes

---

## Code Guide

### The `agent` module

Contains all RL agents. Each agent must implement at least:

* `take_action(observation)`
* `update()` (optional for non-learning agents)

See `random_agent.py` for a minimal example in continuous environments.

### The `world` module

Includes all simulation logic, visualization, and file I/O for both grid and continuous modes.

#### `grid_creator.py`

Run this file to design grid-worlds in your browser:

```bash
python grid_creator.py
```

Go to [127.0.0.1:5000](http://127.0.0.1:5000) and design maps using interactive tools. Saved files appear in `grid_configs/`.

#### `continuous_space_creator_web.py`

Run this to design restaurant environments in continuous 2D space:

```bash
python continuous_space_creator_web.py
```

Then visit [127.0.0.1:5000](http://127.0.0.1:5000) to create your layout.

#### `environment.py`

Contains two main environments:

* `Environment`: for discrete grid-worlds
* `ContinuousEnvironment`: for continuous delivery tasks with sensors and physics

Both provide:

* `reset()` to start a new episode
* `step(action)` to take an action and get the next state
* `evaluate_agent()` to test trained policies

#### `grid.py` and `continuous_space.py`

These represent the world in memory:

* `Grid`: discrete 2D arrays
* `ContinuousSpace`: width, height, table radius, and table locations

<!-- #### `gui.py`

Handles visualizations via PyGame. Works for both environments. GUI can be turned off with `--no_gui` during training for speed. -->