"""
Continuous Space Creator (Web)

Allows interactive placement of tables on a continuous map
using a browser interface, similar to grid_creator.
"""

import ast
from flask import Flask, render_template, request
from flask_socketio import SocketIO
import numpy as np
from pathlib import Path

# Import ContinuousSpace
try:
    from world.continuous_space import ContinuousSpace
    from world import GRID_CONFIGS_FP  # reuse grid config directory
except ModuleNotFoundError:
    import sys
    from os import path, pardir
    root_path = path.abspath(path.join(path.join(path.abspath(__file__), pardir), pardir))
    if root_path not in sys.path:
        sys.path.append(root_path)

    from world.continuous_space import ContinuousSpace
    from world import GRID_CONFIGS_FP

# Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socket_io = SocketIO(app)

@app.route('/')
def home():
    return render_template("restaurant_editor_pickup.html")

@app.route('/build_space')
def build_space():
    """
    Request params:
        width: float
        height: float
        table_radius: float
        tables: list of (x, y) positions
        pickup_point: (x, y)
        name: filename to save under
        save: 'true' or 'false'
    """
    width = float(request.args.get("width"))
    height = float(request.args.get("height"))
    table_radius = float(request.args.get("table_radius"))
    tables = ast.literal_eval(request.args.get("tables"))
    pickup = request.args.get("pickup_point")
    name = str(request.args.get("name"))
    to_save = request.args.get("save", "false").lower() == "true"

    tables_np = [np.array(t) for t in tables]
    pickup_point = np.array(ast.literal_eval(pickup)) if pickup else None

    space = ContinuousSpace(width, height, tables_np, table_radius)

    if to_save and len(name) > 0:
        save_fp = GRID_CONFIGS_FP / f"{name}.npz"
        space.save(save_fp)
        return {
            "success": "true",
            "save_fp": str(save_fp)
        }

    return {
        "success": "false",
        "message": "Environment created but not saved."
    }

if __name__ == '__main__':
    socket_io.run(app, debug=True, allow_unsafe_werkzeug=True)
