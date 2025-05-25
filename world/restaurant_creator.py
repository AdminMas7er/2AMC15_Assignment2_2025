# continuous_space_creator.py

import numpy as np
from pathlib import Path
from continuous_space import ContinuousSpace
import argparse
import random


def generate_space(width=10.0, height=10.0, table_radius=0.5, n_tables=5, seed=42):
    random.seed(seed)
    tables = []
    max_attempts = 1000
    attempts = 0

    while len(tables) < n_tables and attempts < max_attempts:
        x = random.uniform(table_radius, width - table_radius)
        y = random.uniform(table_radius, height - table_radius)
        pos = np.array([x, y])

        if all(np.linalg.norm(pos - t) >= 2 * table_radius for t in tables):
            tables.append(pos)

        attempts += 1

    if len(tables) < n_tables:
        raise ValueError("Could not place all tables without overlap.")

    return ContinuousSpace(width, height, tables, table_radius)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a continuous delivery space.")
    parser.add_argument("--width", type=float, default=10.0)
    parser.add_argument("--height", type=float, default=10.0)
    parser.add_argument("--radius", type=float, default=0.5, help="Table radius")
    parser.add_argument("--count", type=int, default=5, help="Number of tables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True, help="Output .npz file name")

    args = parser.parse_args()
    space = generate_space(args.width, args.height, args.radius, args.count, args.seed)
    space.save(Path(args.output))
    print(f"Saved continuous space to {args.output}.npz")
