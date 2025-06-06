# continuous_space.py

import numpy as np
from pathlib import Path

class ContinuousSpace:
    def __init__(self, width: float, height: float, tables: list[np.ndarray], table_radius: float, pickup_point: np.ndarray = None):
        self.width = width
        self.height = height
        self.tables = tables
        self.table_radius = table_radius
        self.pickup_point = pickup_point if pickup_point is not None else np.array([width / 2, height / 2])


    def to_dict(self):
        return {
            "width": self.width,
            "height": self.height,
            "table_radius": self.table_radius,
            "tables": np.array(self.tables),
            "pickup_point": self.pickup_point,
        }

    def save(self, fp: Path):
        data = self.to_dict()
        np.savez(fp.with_suffix(".npz"), **data)

    @staticmethod
    def load(fp: Path) -> "ContinuousSpace":
        data = np.load(fp.with_suffix(".npz"), allow_pickle=True)
        width = float(data["width"])
        height = float(data["height"])
        table_radius = float(data["table_radius"])
        tables = data["tables"]
        pickup_point = data["pickup_point"] if "pickup_point" in data else np.array([width / 2, height / 2])
        return ContinuousSpace(width, height, list(tables), table_radius, pickup_point)
