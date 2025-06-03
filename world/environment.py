import numpy as np
import math
import random
import pygame
from pygame import gfxdraw
from pathlib import Path
from world.continuous_space import ContinuousSpace  # if not already imported

WINDOW_SIZE = (800, 600)
TABLE_COLOR = (139, 69, 19)
AGENT_COLOR = (0, 102, 204)
BACKGROUND_COLOR = (255, 255, 255)
SENSOR_COLOR = (150, 150, 150)

class ContinuousEnvironment:
    def __init__(self, width=10.0, height=10.0, table_radius=0.5, n_tables=3, seed=42, enable_gui=True, space_file: Path = None):
        if space_file is not None:
            # Load from saved file
            space = ContinuousSpace.load(space_file)
            self.width = space.width
            self.height = space.height
            self.tables = space.tables
            self.table_radius = space.table_radius
            self.pickup_point = space.pickup_point
            self.has_order = False
            self.current_target_table = None

        else:
            # Use defaults (for random generation/testing)
            self.width = width
            self.height = height
            self.table_radius = table_radius
            self.tables = self._generate_tables(n_tables, seed)

        self.agent_pos = np.array([1.0, 1.0])
        self.agent_angle = 0.0
        self.sensor_angles = [0, -np.radians(30), np.radians(30)]
        self.max_sensor_range = 5.0
        self.enable_gui = enable_gui
        self.window = None
        self.screen_scale = 50

        if self.enable_gui:
            import pygame
            pygame.init()
            self.window = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("Continuous Environment")

        self.reset()

    def _generate_tables(self, n_tables, seed):
        random.seed(seed)
        tables = []
        for _ in range(self.n_tables):
            x = self.random.uniform(self.table_radius, self.width - self.table_radius)
            y = self.random.uniform(self.table_radius, self.height - self.table_radius)
            tables.append(np.array([x, y]))
        return tables

    def reset(self, pos=(1.0, 1.0), angle=0.0):
        self.agent_pos = np.array(pos)
        self.agent_angle = angle
        self.has_order = False
        self.current_target_table = None
        return self._get_observation()

    def step(self, action):
        velocity, rotation = action
        self.agent_angle += rotation
        dx = velocity * math.cos(self.agent_angle)
        dy = velocity * math.sin(self.agent_angle)
        new_pos = self.agent_pos + np.array([dx, dy])

        # Make the move if the position is not in a table and inside the map
        if self._is_valid_position(new_pos):
            self.agent_pos = new_pos

        obs = self._get_observation()
        if self.enable_gui:
            self._render(obs)

        reward = -0.01  # Small negative step cost
        done = False

        # If agent reaches the pickup point
        if not self.has_order and np.linalg.norm(self.agent_pos - self.pickup_point) < 0.3:
            self.has_order = True
            self.current_target_table = random.choice(self.tables)
            reward = 1.0  # Reward for picking up

        # If agent reaches the delivery table
        elif self.has_order and np.linalg.norm(self.agent_pos - self.current_target_table) < self.table_radius:
            self.has_order = False
            self.current_target_table = None
            reward = 5.0  # Reward for delivery
            done = True  # Optional

        obs = self._get_observation()
        if self.enable_gui:
            self._render(obs)

        return obs, reward, done


    def _is_valid_position(self, pos):
        """Collission detection for whether the agent walks into a table or outside of the map,
        Currently does not keep track of the radius of the agent itself since this has not been chosen yet"""
        x, y = pos
        if not (0 <= x <= self.width and 0 <= y <= self.height):
            return False
        for table in self.tables:
            if np.linalg.norm(pos - table) < self.table_radius:
                return False
        return True

    def _get_sensor_distance(self, angle_offset):
        angle = self.agent_angle + angle_offset
        for dist in np.linspace(0, self.max_sensor_range, num=100):
            probe = self.agent_pos + dist * np.array([math.cos(angle), math.sin(angle)])
            if not self._is_valid_position(probe):
                return dist
        return self.max_sensor_range

    def _get_observation(self):
        distances = [self._get_sensor_distance(a) for a in self.sensor_angles]
        return {
            "agent_pos": self.agent_pos.copy(),
            "agent_angle": self.agent_angle,
            "sensor_distances": distances,
            "target_tables": self.tables.copy(),
            "pickup_point": self.pickup_point,
            "has_order": self.has_order,
            "current_target_table": self.current_target_table,
        }

    def _render(self, obs):
        self.window.fill(BACKGROUND_COLOR)

        def to_px(pos):
            return int(pos[0] * self.screen_scale), int(self.height * self.screen_scale - pos[1] * self.screen_scale)

        # Draw tables
        for table in obs["target_tables"]:
            pygame.draw.circle(
                self.window, TABLE_COLOR,
                to_px(table), int(self.table_radius * self.screen_scale)
            )

        # Draw pickup point
        pygame.draw.circle(self.window, (0, 255, 0), to_px(self.pickup_point), 8)

        # Draw agent
        agent_px = to_px(obs["agent_pos"])
        pygame.draw.circle(self.window, AGENT_COLOR, agent_px, 10)

        # Draw sensors
        for i, angle in enumerate(self.sensor_angles):
            sensor_distance = obs["sensor_distances"][i]
            sensor_end = obs["agent_pos"] + sensor_distance * np.array([
                math.cos(obs["agent_angle"] + angle),
                math.sin(obs["agent_angle"] + angle)
            ])
            pygame.draw.line(
                self.window, SENSOR_COLOR,
                agent_px, to_px(sensor_end), 1
            )

        pygame.display.flip()
        pygame.time.delay(50)

    @staticmethod
    def _default_reward_function(obs):
        """A simple reward function that gives a positive reward for being close to the table which got an order."""
        

    def close(self):
        if self.enable_gui:
            pygame.quit()
