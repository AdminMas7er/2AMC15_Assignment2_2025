import numpy as np
import math
import time
from pathlib import Path
import pygame
from world.continuous_space import ContinuousSpace 


# ------------------------------
#  Global visual parameters
# ------------------------------
WINDOW_SIZE = (800, 600)
TABLE_COLOR = (139, 69, 19)
AGENT_COLOR = (0, 102, 204)
BACKGROUND_COLOR = (255, 255, 255)

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
            self.screen_scale = min(WINDOW_SIZE[0] / self.width,
                            WINDOW_SIZE[1] / self.height)

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

        self.steps_taken += 1
        self.cumulative_reward += reward

        obs = self._get_observation()
        if self.enable_gui:
            self._render(obs)

        return obs, reward, done, {"collision": collision}

    # ------------------------------------------------------------------
    def _is_valid_position(self, pos):
        """True if *pos* is inside the map and not colliding with a table."""
        x, y = pos
        if not (0 <= x <= self.width and 0 <= y <= self.height):
            return False
        for table in self.tables:
            if np.linalg.norm(pos - table) < self.table_radius:
                return False
        return True

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

    # GUI------------------------------------------------------------------
    def _render(self, obs):
        font = pygame.font.SysFont("Arial", 16)
        self.window.fill(BACKGROUND_COLOR)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()

        def to_px(p):
            return (int(p[0] * self.screen_scale),
                    int(self.height * self.screen_scale - p[1] * self.screen_scale))

        # tables
        for idx, tbl in enumerate(self.tables):
            color = (255, 0, 0) if idx == obs["current_target"] else TABLE_COLOR
            pygame.draw.circle(self.window, color, to_px(tbl),
                               int(self.table_radius * self.screen_scale))
            lbl = font.render(str(idx + 1), True, (255, 255, 255))
            self.window.blit(lbl, lbl.get_rect(center=to_px(tbl)))
        
        #pickup (kitchen)
        pickup_px = to_px(self.pickup_point)
        rect_width = 80
        rect_height = 40
        pickup_rect = pygame.Rect(
            pickup_px[0] - rect_width // 2,
            pickup_px[1] - rect_height // 2,
            rect_width,
            rect_height
        )
        pygame.draw.rect(self.window, (0, 200, 0), pickup_rect)
        font = pygame.font.SysFont("Arial", 16)
        label = font.render("KITCHEN", True, (255, 255, 255))  # White text
        label_rect = label.get_rect(center=pickup_rect.center)
        self.window.blit(label, label_rect)
        

        # agent
        pygame.draw.circle(self.window, AGENT_COLOR, to_px(obs["agent_pos"]), 10)

        # HUD
        panel_w = 200
        pygame.draw.rect(self.window, (240, 240, 240),
                         pygame.Rect(WINDOW_SIZE[0]-panel_w, 0, panel_w, 100))
        elapsed = time.time() - self.episode_start_time
        def text(t, y): self.window.blit(font.render(t, True, (0,0,0)),
                                         (WINDOW_SIZE[0]-panel_w+10, y))
        text(f"Time: {elapsed:4.1f}s", 10)
        text(f"Steps:{self.steps_taken:4d}", 30)
        text(f"Reward:{self.cumulative_reward:6.1f}", 50)
        text(f"Target:{obs['current_target']+1}", 70)

        pygame.display.flip()
        pygame.time.delay(50)

    #  close
    def close(self):
        if getattr(self, "enable_gui", False):
            pygame.quit()

    #  properties
    @property
    def state_size(self):
        return 2 + 2 + len(self.tables)

    @property
    def action_size(self):
        return len(self.actions)
