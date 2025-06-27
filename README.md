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

### train_sac.py usage for continuous space:

```bash
usage: train_sac.py --restaurant RESTAURANT [options]

Train a Soft Actor-Critic (SAC) agent in a continuous restaurant environment.

options:
  --restaurant RESTAURANT     Path to the .npz restaurant layout file (default: grid_configs/my_first_restaurant.npz)
  --episodes EPISODES         Number of training episodes (default: 2000)
  --max_steps MAX_STEPS       Maximum steps per episode (default: 500)
  --eval_freq EVAL_FREQ       Evaluation frequency in episodes (default: 100)
  --eval_episodes N           Number of evaluation episodes per evaluation (default: 10)
  --save_freq SAVE_FREQ       Model saving frequency in episodes (default: 500)
  --no_gui                    Disable GUI rendering during training
  --render_eval               Render GUI during evaluation episodes
  --seed SEED                 Random seed (default: 42)
  --output_dir DIR            Output directory for logs and model checkpoints (default: results)
  --load_model PATH           Path to pre-trained model to resume training or demo

SAC-specific hyperparameters:
  --lr LR                     Learning rate (default: 1e-3)
  --gamma GAMMA               Discount factor (default: 0.98)
  --tau TAU                   Soft target update coefficient (default: 0.01)
  --alpha ALPHA               Initial entropy temperature (default: 0.5)
  --auto_temp                 Enable automatic temperature tuning (default: enabled)
  --buffer_size SIZE          Replay buffer size (default: 50000)
  --batch_size SIZE           Batch size for training (default: 128)
  --hidden_size SIZE          Hidden layer size for networks (default: 512)
```

<!-- 
## Code Guide

### The `agent` module

Contains all RL agents. Each agent must implement at least:

* `take_action(observation)`
* `update()` (optional for non-learning agents)

See `random_agent.py` for a minimal example in continuous environments.

### The `world` module

Includes all simulation logic, visualization, and file I/O for both the continuous maps.

#### `continuous_space_creator_web.py`

Run this to design restaurant environments in continuous 2D space:

```bash
python continuous_space_creator_web.py
```

Then visit [127.0.0.1:5000](http://127.0.0.1:5000) to create your layout. Saved files appear in `grid_configs/`.

#### `environment.py`

Contains two main environments:

* `Environment`: for discrete grid-worlds
* `ContinuousEnvironment`: for continuous delivery tasks with sensors and physics

Both provide:

* `reset()` to start a new episode
* `step(action)` to take an action and get the next state
* `evaluate_agent()` to test trained policies

#### `continuous_space.py`

These represent the world in memory:

* `ContinuousSpace`: width, height, table radius, and table locations -->

<!-- #### `gui.py`

Handles visualizations via PyGame. Works for both environments. GUI can be turned off with `--no_gui` during training for speed. -->