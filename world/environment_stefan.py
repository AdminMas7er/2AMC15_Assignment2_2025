import numpy as np
import math
import random
import pygame
import time
from pygame import gfxdraw
from pathlib import Path
from world.continuous_space import ContinuousSpace  # if not already imported

WINDOW_SIZE = (800, 600)
TABLE_COLOR = (139, 69, 19)
AGENT_COLOR = (0, 102, 204)
BACKGROUND_COLOR = (255, 255, 255)
SENSOR_COLOR = (150, 150, 150)

# Pre-defined step size and rotation
STEP_LENGTH = 0.1
STEP_ROTATION = np.radians(30)

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

        self.actions = {
            0: (0.0, 0.0), # Do nothing
            1: (0.0, STEP_ROTATION), # Rotate right
            2: (0.0, -STEP_ROTATION), # Rotate left
            3: (STEP_LENGTH, 0.0), # Step forward
            4: (-STEP_LENGTH, 0.0) # Step backward
        }

        self.episode_start_time = time.time()
        self.steps_taken = 0
        self.cumulative_reward = 0.0

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
        self.episode_start_time = time.time()
        self.steps_taken = 0
        self.cumulative_reward = 0.0
        return self._get_observation()

    def step(self, action):
        velocity, rotation = self.actions.get(action) # Get the action from the key
        self.agent_angle += rotation
        dx = velocity * math.cos(self.agent_angle)
        dy = velocity * math.sin(self.agent_angle)
        new_pos = self.agent_pos + np.array([dx, dy])

        # Make the move if the position is not in a table and inside the map
        if self._is_valid_position(new_pos):
            self.agent_pos = new_pos
        else:
            # Negative reward for trying to walk out of bounds
            reward = -5

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
        elif self.has_order and np.linalg.norm(self.agent_pos - self.current_target_table) < self.table_radius * 1.5:
            self.has_order = False
            self.current_target_table = None
            reward = 5.0  # Reward for delivery
            done = True  # Optional

        self.steps_taken += 1
        self.cumulative_reward += reward

        obs = self._get_observation()
        if self.enable_gui:
            self._render(obs)

        return obs, reward, done, {}


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
        font = pygame.font.SysFont("Arial", 16)

        self.window.fill(BACKGROUND_COLOR)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        def to_px(pos):
            return int(pos[0] * self.screen_scale), int(self.height * self.screen_scale - pos[1] * self.screen_scale)

        # Draw tables
        for idx, table in enumerate(obs["target_tables"]):
            px_pos = to_px(table)
            pygame.draw.circle(
                self.window, TABLE_COLOR,
                px_pos, int(self.table_radius * self.screen_scale)
            )

            # Draw table number at center
            label = font.render(str(idx + 1), True, (255, 255, 255))  # White text
            label_rect = label.get_rect(center=px_pos)
            self.window.blit(label, label_rect)

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

        # Draw metrics on the right
        panel_width = 200
        panel_rect = pygame.Rect(WINDOW_SIZE[0] - panel_width, 0, panel_width, 100)
        pygame.draw.rect(self.window, (240, 240, 240), panel_rect)

        elapsed_time = time.time() - self.episode_start_time

        def draw_text(text, x, y):
            label = font.render(text, True, (0, 0, 0))
            self.window.blit(label, (x, y))

        draw_text(f"Time: {elapsed_time:.1f}s", WINDOW_SIZE[0] - panel_width + 10, 10)
        draw_text(f"Steps: {self.steps_taken}", WINDOW_SIZE[0] - panel_width + 10, 30)
        draw_text(f"Reward: {self.cumulative_reward:.2f}", WINDOW_SIZE[0] - panel_width + 10, 50)


        pygame.display.flip()
        pygame.time.delay(50)

    @staticmethod
    def _default_reward_function(obs):
        """A simple reward function that gives a positive reward for being close to the table which got an order."""
        

    def close(self):
        if self.enable_gui:
            pygame.quit()

    #ADDED THIS 12/06
    def get_state_size(self):
        # agent_pos (2), agent_angle (1), sensor_distances (3), pickup_point (2), has_order (1), current_target_table (2)
        return 11

    def get_action_size(self):
        return len(self.actions)