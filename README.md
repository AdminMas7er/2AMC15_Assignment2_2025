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

### Quick Start

```bash
# Basic training
python train_sac.py --episodes 100 --max_steps 300 --eval_freq 25 --no_gui

# Training with visualization
python train_sac.py --episodes 100 --max_steps 300 --eval_freq 25

# Test trained model
python train_sac.py --load_model results/sac_training_TIMESTAMP/models/best_model.pth --episodes 10
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

### Results

Training results are saved in `results/sac_training_TIMESTAMP/` with models, plots, and logs. Key metrics include success rate, mean reward, and episode length.



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