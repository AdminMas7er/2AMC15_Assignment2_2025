import numpy as np
import math
<<<<<<< HEAD
import random
import pygame
import time
from pygame import gfxdraw
from pathlib import Path
from world.continuous_space import ContinuousSpace  # if not already imported

=======
import time
from pathlib import Path
import pygame
from world.continuous_space import ContinuousSpace 

# ------------------------------
#  Global visual parameters
# ------------------------------
>>>>>>> origin/DQN_final
WINDOW_SIZE = (800, 600)
TABLE_COLOR = (139, 69, 19)
AGENT_COLOR = (0, 102, 204)
BACKGROUND_COLOR = (255, 255, 255)
<<<<<<< HEAD
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
        self.deliveries_done=0
        self.current_episode=0
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
    def set_episode(self, episode):
        self.current_episode = episode
    def reset(self, pos=None, angle=0.0):
        
        if pos is None:
            pos = self.pickup_point.copy()
        self.agent_pos = np.array(pos)
        self.agent_angle = angle

        #Starting with the order already assigned (skipping pickup phase)
        self.has_order = True #starts with order
        self.current_target_table = random.choice(self.tables) #later we can bring this back once it works for a fixed table
        # self.current_target_table = self.tables[2] #Fixed target table to learn quicker

        self.episode_start_time = time.time()
        self.steps_taken = 0
        self.cumulative_reward = 0.0
        print("Resetting agent_pos to pickup_point:", self.agent_pos, "pickup_point:", self.pickup_point)

        # Initialize visit counter
        self.visit_counter = {}
        #skibidi test
        return self._get_observation()
    def get_no_deliveries(self):
        """Returns the number of successful deliveries made by the agent."""
        return self.deliveries_done
    def step(self, action):
        velocity, rotation = self.actions.get(action) # Get the action from the key
        self.agent_angle += rotation
        dx = velocity * math.cos(self.agent_angle)
        dy = velocity * math.sin(self.agent_angle)
        new_pos = self.agent_pos + np.array([dx, dy])
        # Make the move if the position is not in a table and inside the map
        reward = -0.01  # Small negative step cost
        done = False
        if self._is_valid_position(new_pos):
            self.agent_pos = new_pos
        else:
            # Negative reward for trying to walk out of bounds
            reward = -5

        obs = self._get_observation()
        if self.enable_gui:
            self._render(obs)

      

        # # If agent reaches the pickup point
        # if not self.has_order and np.linalg.norm(self.agent_pos - self.pickup_point) < 0.3:
        #     self.has_order = True
        #     self.current_target_table = random.choice(self.tables)
        #     reward = 1.0  # Reward for picking up

        # If agent reaches the delivery table
        if self.has_order and np.linalg.norm(self.agent_pos - self.current_target_table) < self.table_radius * 1.5:
            print("Agent reached the target table:", self.current_target_table)
            self.deliveries_done += 1
            self.has_order = False
            self.current_target_table = None
            reward = 20.0  # Reward for delivery
            done = True  # Optional

        cell_size = 0.5
        cell = tuple((self.agent_pos / cell_size).astype(int))
        self.visit_counter[cell] = self.visit_counter.get(cell, 0) + 1
        visit_penalty = -0.01 * (1.05 ** self.visit_counter[cell]) #Penalty for camping in the same spot
        reward += visit_penalty
=======

# ------------------------------
#  Movement parameters
# ------------------------------
STEP_LENGTH = 0.5                     
STEP_ROTATION = np.radians(30)        
SUCCESS_RADIUS = STEP_LENGTH * 1.5     
TIME_PENALTY   = -0.05                 
COLLISION_PEN  = -3.0                  


class ContinuousEnvironment:
    """Continuous 2‑D restaurant delivery environment.

    The agent only observes its own (x, y) position, heading (cosθ, sinθ),
    whether it currently carries an order, and the index of the target table.
    It *never* sees the absolute coordinates of any table. This satisfies the
    course requirement that the agent may only use self‑information.
    """

    # ------------------------------------------------------------------
    def __init__(self, width: float = 10.0, height: float = 10.0,
                 table_radius: float = 0.5, n_tables: int = 3, seed: int = 42,
                 enable_gui: bool = True, space_file: Path | None = None):
        """Create the environment.

        Args:
            width/height: physical size in metres.
            table_radius: radius of every table.
            n_tables:     how many tables to randomly place when *space_file* is None.
            seed:         PRNG seed for reproducibility.
            enable_gui:   render with Pygame if *True*.
            space_file:   optional *.npz* layout created elsewhere.
        """
        self.np_random = np.random.RandomState(seed)

        if space_file is not None:
            space = ContinuousSpace.load(space_file)
            self.width, self.height = space.width, space.height
            self.tables            = space.tables
            self.table_radius      = space.table_radius
            self.pickup_point      = space.pickup_point
        else:
            self.width, self.height = width, height
            self.table_radius       = table_radius
            self.tables             = self._generate_tables(n_tables)
            self.pickup_point       = np.array([width / 2, height / 2])

        # ---------------- agent state ----------------
        self.agent_pos   = np.array([1.0, 1.0])     
        self.agent_angle = 0.0                     

        # delivery counter
        self.deliveries_done = 0

        # action mapping: (forward/backwards velocity, rotation delta)
        self.actions: dict[int, tuple[float, float]] = {
            0: (0.0, 0.0),                # do nothing
            1: (0.0,  STEP_ROTATION),     # rotate left
            2: (0.0, -STEP_ROTATION),     # rotate right
            3: ( STEP_LENGTH, 0.0),       # step forward
            4: (-STEP_LENGTH, 0.0)        # step backward
        }

        # GUI -----------------------------------------------------------
        self.enable_gui = enable_gui
        if self.enable_gui:
            pygame.init()
            self.window = pygame.display.set_mode(WINDOW_SIZE)
            pygame.display.set_caption("Continuous Environment")
            self.screen_scale = 80        # pixels per metre

        self.reset()

    # ------------------------------------------------------------------
    def _generate_tables(self, n_tables: int):
        """Random table placement, ensuring they stay inside the map."""
        tables = []
        for _ in range(n_tables):
            x = self.np_random.uniform(self.table_radius,
                                       self.width - self.table_radius)
            y = self.np_random.uniform(self.table_radius,
                                       self.height - self.table_radius)
            tables.append(np.array([x, y], dtype=np.float32))
        return tables

    # ------------------------------------------------------------------
    def reset(self, pos=None, angle: float = 0.0):
        """Start a new episode."""
        self.agent_pos   = np.array(pos) if pos is not None else self.pickup_point.copy()
        self.agent_angle = angle
        self.has_order   = True
        self.current_target_id = self.np_random.randint(len(self.tables))

        self.steps_taken = 0
        self.cumulative_reward = 0.0
        self.visit_count: dict[tuple[int,int], int] = {}
        self.episode_start_time = time.time()

        return self._get_observation()

    # ------------------------------------------------------------------
    def step(self, action: int):
        """Advance the simulation by one time‑step."""
        v, dtheta = self.actions[action]
        self.agent_angle += dtheta

        dx = v * math.cos(self.agent_angle)
        dy = v * math.sin(self.agent_angle)
        new_pos = self.agent_pos + np.array([dx, dy])

        reward = TIME_PENALTY        # base time cost
        done   = False
        collision = False

        # ----- collision check -----
        if not self._is_valid_position(new_pos):
            reward += COLLISION_PEN
            collision = True
        else:
            self.agent_pos = new_pos

        # ----- success check -------
        tgt_pos = self.tables[self.current_target_id]
        if self.has_order and np.linalg.norm(self.agent_pos - tgt_pos) < SUCCESS_RADIUS:
            reward += 50.0
            done = True
            self.has_order = False
            self.deliveries_done += 1

        # ----- exploration bonus ---
        cell = tuple((self.agent_pos / 0.75).astype(int))
        if self.steps_taken < 600 and cell not in self.visit_count:
            self.visit_count[cell] = 1
            reward += 0.5

        # small turning penalty to discourage spinning
        if action in (1, 2):
            reward -= 0.005

>>>>>>> origin/DQN_final
        self.steps_taken += 1
        self.cumulative_reward += reward

        obs = self._get_observation()
        if self.enable_gui:
            self._render(obs)

<<<<<<< HEAD
        return obs, reward, done, {}


    def _is_valid_position(self, pos):
        """Collission detection for whether the agent walks into a table or outside of the map,
        Currently does not keep track of the radius of the agent itself since this has not been chosen yet"""
=======
        return obs, reward, done, {"collision": collision}

    # ------------------------------------------------------------------
    def _is_valid_position(self, pos):
        """True if *pos* is inside the map and not colliding with a table."""
>>>>>>> origin/DQN_final
        x, y = pos
        if not (0 <= x <= self.width and 0 <= y <= self.height):
            return False
        for table in self.tables:
            if np.linalg.norm(pos - table) < self.table_radius:
                return False
        return True

<<<<<<< HEAD
    def _get_sensor_distance(self, angle_offset):
        angle = self.agent_angle + angle_offset
        for dist in np.linspace(0, self.max_sensor_range, num=100):
            probe = self.agent_pos + dist * np.array([math.cos(angle), math.sin(angle)])
            if not self._is_valid_position(probe):
                return dist
        return self.max_sensor_range

    def _get_observation(self):
        return {
            "agent_pos": self.agent_pos.copy(),
            "target_tables": self.tables.copy(), 
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

            # Determine color:
            if obs["current_target_table"] is not None and np.allclose(table, obs["current_target_table"], atol=1e-2):
                color = (255, 0, 0)  # Red for current target table
            else:
                color = TABLE_COLOR  # Normal table color

            # Draw table circle
            pygame.draw.circle(
                self.window, color,
                px_pos, int(self.table_radius * self.screen_scale)
            )

            # Draw table number at center
            label = font.render(str(idx + 1), True, (255, 255, 255))  # White text
            label_rect = label.get_rect(center=px_pos)
            self.window.blit(label, label_rect)

        # Draw pickup point
        # Draw pickup point as rectangle with "KITCHEN" label
        pickup_px = to_px(self.pickup_point)
        rect_width = 80
        rect_height = 40

        # Rectangle centered at pickup point
        pickup_rect = pygame.Rect(
            pickup_px[0] - rect_width // 2,
            pickup_px[1] - rect_height // 2,
            rect_width,
            rect_height
        )

        # Draw rectangle (green)
        pygame.draw.rect(self.window, (0, 200, 0), pickup_rect)

        # Draw "KITCHEN" label
        font = pygame.font.SysFont("Arial", 16)
        label = font.render("KITCHEN", True, (255, 255, 255))  # White text
        label_rect = label.get_rect(center=pickup_rect.center)
        self.window.blit(label, label_rect)

        # Draw agent
        agent_px = to_px(obs["agent_pos"])
        pygame.draw.circle(self.window, AGENT_COLOR, agent_px, 10)

        # # Draw sensors
        # for i, angle in enumerate(self.sensor_angles):
        #     sensor_distance = obs["sensor_distances"][i]
        #     sensor_end = obs["agent_pos"] + sensor_distance * np.array([
        #         math.cos(obs["agent_angle"] + angle),
        #         math.sin(obs["agent_angle"] + angle)
        #     ])
        #     pygame.draw.line(
        #         self.window, SENSOR_COLOR,
        #         agent_px, to_px(sensor_end), 1
        #     )

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

        #Add the target table in the legend
        if obs["current_target_table"] is None:
            target_table_text = "Target table = None"
        else:
            target_idx = None
            for idx, table in enumerate(obs["target_tables"]):
                if np.allclose(table, obs["current_target_table"], atol=1e-2):
                    target_idx = idx + 1  # 1-based index
                    break
            target_table_text = f"Target table = {target_idx}"

        # Draw it below reward
        draw_text(target_table_text, WINDOW_SIZE[0] - panel_width + 10, 70)

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
        n_tables = len(self.tables)
        # State vector: agent position (2), target table one-hot encoding (n_tables)
        return 2 + n_tables

    def get_action_size(self):
        return len(self.actions)
=======
    # ------------------------------------------------------------------
    def _get_observation(self):
        """Return the minimal agent‑centric observation."""
        heading = np.array([math.cos(self.agent_angle),
                            math.sin(self.agent_angle)], dtype=np.float32)
        return {
            "agent_pos":      self.agent_pos.copy(),   
            "heading":        heading,                 # (cosθ, sinθ)
            "current_target": self.current_target_id,  
            # "has_order":      self.has_order          
        }

    # ------------------------------------------------------------------
    # GUI rendering left unchanged – omitted here for brevity
    # ------------------------------------------------------------------

    @property
    def state_size(self):
        """Length of the flattened observation vector used by the agent."""
        return 2 + 2 + len(self.tables)  

    @property
    def action_size(self):
        return len(self.actions)
>>>>>>> origin/DQN_final
