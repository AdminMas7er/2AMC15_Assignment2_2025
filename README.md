Here’s an updated version of your README to incorporate the new continuous-space environment, while keeping the structure and tone closely aligned with the original:

---

# Welcome to Data Intelligence Challenge-2AMC15!

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
   `python continuous_space_creator_web.py`
   Then go to [127.0.0.1:5000](http://127.0.0.1:5000) or http://localhost:5000 to place tables and export your layout as `.npz`.

2. Run training or testing on the saved `.npz` space:

   ```bash
   python train.py --agent random_agent --restaurant grid_configs/my_first_restaurant.npz --iter 500
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

#### `gui.py`

Handles visualizations via PyGame. Works for both environments. GUI can be turned off with `--no_gui` during training for speed.