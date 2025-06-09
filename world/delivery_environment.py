import numpy as np
import math
import random
import pygame
from pygame import gfxdraw
from pathlib import Path
from world.continuous_space import ContinuousSpace

class DeliveryEnvironment:
    """
    Deep Reinforcement Learning Environment for Delivery Robot
    Continuous state space, discrete action space
    """
    def __init__(self, space_file: Path = None, enable_gui=True, max_steps=500):
        if space_file is not None:
            space = ContinuousSpace.load(space_file)
            self.width = space.width
            self.height = space.height
            self.tables = space.tables
            self.table_radius = space.table_radius
        else:
            # Default configuration
            self.width = 10.0
            self.height = 10.0
            self.table_radius = 0.5
            self.tables = [np.array([3.0, 3.0]), np.array([7.0, 7.0])]

        # Agent parameters
        self.agent_radius = 0.2
        self.max_speed = 0.3
        self.max_angular_speed = np.radians(45)
        
        # Sensor configuration
        self.sensor_angles = [0, -np.radians(45), np.radians(45), 
                             -np.radians(90), np.radians(90)]
        self.max_sensor_range = 3.0
        
        # Environment state
        self.agent_pos = np.array([1.0, 1.0])
        self.agent_angle = 0.0
        self.target_table_idx = 0
        self.step_count = 0
        self.max_steps = max_steps
        self.done = False
        
        # Action space (discrete)
        self.actions = {
            0: (0.0, 0.0),                    # Stop
            1: (self.max_speed, 0.0),         # Forward
            2: (-self.max_speed/2, 0.0),      # Backward
            3: (0.0, self.max_angular_speed), # Turn left
            4: (0.0, -self.max_angular_speed),# Turn right
            5: (self.max_speed, self.max_angular_speed),  # Forward + left
            6: (self.max_speed, -self.max_angular_speed), # Forward + right
        }
        self.action_space_size = len(self.actions)
        
        # GUI settings
        self.enable_gui = enable_gui
        self.window = None
        self.screen_scale = 50
        
        if self.enable_gui:
            pygame.init()
            self.window = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("Delivery Robot Environment")

    def reset(self):
        """Reset environment"""
        # Randomly select starting position (avoid tables)
        while True:
            self.agent_pos = np.array([
                random.uniform(self.agent_radius, self.width - self.agent_radius),
                random.uniform(self.agent_radius, self.height - self.agent_radius)
            ])
            if self._is_valid_position(self.agent_pos):
                break
        
        self.agent_angle = random.uniform(0, 2 * np.pi)
        self.target_table_idx = random.randint(0, len(self.tables) - 1)
        self.step_count = 0
        self.done = False
        
        return self._get_state()

    def step(self, action):
        """Testing different options here, since DQN is very sensitive to reward function and does not seem to be working great
        OPTION 1: SIMPLE & DEBUGGING: just minimizing the distance to the target
        OPTION 2: DEFAULT AS GIJS PROPOSED
        OPTION 3: DEFAULT AS GIJS PROPOSED, BUT THEN WITH PENALTY FOR ENDLESLY TURNING OR GOING BACKWARDS
        *****ONLY USE ONE OF THEM AT THE TIME TO TRY******
        So far number 1 actually seems to work best.. """

        #****OPTION 1: SIMPLE & DEBUGGING:****
        #************************************************************************************************
        if self.done:
            return self._get_state(), 0, True, {}

        linear_vel, angular_vel = self.actions[action]

        self.agent_angle += angular_vel
        self.agent_angle = self.agent_angle % (2 * np.pi)

        dx = linear_vel * math.cos(self.agent_angle)
        dy = linear_vel * math.sin(self.agent_angle)
        new_pos = self.agent_pos + np.array([dx, dy])

        reward = 0.0

        if not self._is_valid_position(new_pos):
            reward = -10  # collision penalty
        else:
            self.agent_pos = new_pos

            target_pos = self.tables[self.target_table_idx]
            distance_to_target = np.linalg.norm(self.agent_pos - target_pos)

            reward = -distance_to_target  # Negative distance → closer = better

            if distance_to_target < self.table_radius + self.agent_radius + 0.1:
                reward += 100
                self.done = True
                print(f"INFO: Target table reached at step {self.step_count}!")


        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True
            reward -= 20

        state = self._get_state()

        if self.enable_gui:
            self._render()

        #****OPTION 2: DEFAULT AS GIJS PROPOSED ****
       # *************************************************************************************************
       #  if self.done:
       #      return self._get_state(), 0, True, {}
       #
       #  # Execute action
       #  linear_vel, angular_vel = self.actions[action]
       #
       #  # Update agent state
       #  self.agent_angle += angular_vel
       #  self.agent_angle = self.agent_angle % (2 * np.pi)
       #
       #  dx = linear_vel * math.cos(self.agent_angle)
       #  dy = linear_vel * math.sin(self.agent_angle)
       #  new_pos = self.agent_pos + np.array([dx, dy])
       #
       #  reward = -0.01  # Time penalty
       #
       #  # Proceed if valid move:
       #  if self._is_valid_position(new_pos):
       #      self.agent_pos = new_pos
       #
       #      # Check if target reached
       #      target_pos = self.tables[self.target_table_idx]
       #      distance_to_target = np.linalg.norm(self.agent_pos - target_pos)
       #
       #      if distance_to_target < self.table_radius + self.agent_radius + 0.1:
       #          reward += 100  # Target reached reward
       #          self.done = True
       #          print("INFO: Target table reached!")
       #      else:
       #          # Distance reward (encourage approaching target)
       #          reward += max(0, 5.0 - distance_to_target) * 0.1
       #  else:
       #      # Collision penalty
       #      reward -= 10
       #
       #  self.step_count += 1
       #  if self.step_count >= self.max_steps:
       #      self.done = True
       #      reward -= 20  # Timeout penalty
       #
       #  state = self._get_state()
       #
       #  if self.enable_gui:
       #      self._render()

        #****OPTION 3 ****
        #***************************************************************************************************
        # if self.done:
        #     return self._get_state(), 0, True, {}
        #
        # # Execute action
        # linear_vel, angular_vel = self.actions[action]
        #
        # # Update agent state
        # self.agent_angle += angular_vel
        # self.agent_angle = self.agent_angle % (2 * np.pi)
        #
        # dx = linear_vel * math.cos(self.agent_angle)
        # dy = linear_vel * math.sin(self.agent_angle)
        # new_pos = self.agent_pos + np.array([dx, dy])
        #
        # reward = -0.01  # Time penalty
        #
        # # Penalize backward movement (action 2)
        # if action == 2:
        #     reward -= 0.1  # discourage moving backward
        #
        # # Penalize pure rotations (actions 3 and 4)
        # if action in [3, 4]:
        #     reward -= 0.05  # discourage endless turning
        #
        # # Proceed if valid move:
        # if self._is_valid_position(new_pos):
        #     self.agent_pos = new_pos
        #
        #     # Check if target reached
        #     target_pos = self.tables[self.target_table_idx]
        #     distance_to_target = np.linalg.norm(self.agent_pos - target_pos)
        #
        #     if distance_to_target < self.table_radius + self.agent_radius + 0.1:
        #         reward += 100  # Target reached reward
        #         self.done = True
        #         print("INFO: Target table reached!")
        #     else:
        #         # Distance reward (encourage approaching target)
        #         reward += max(0, 5.0 - distance_to_target) * 0.1
        # else:
        #     # Collision penalty
        #     reward -= 10
        #
        # self.step_count += 1
        # if self.step_count >= self.max_steps:
        #     self.done = True
        #     reward -= 20  # Timeout penalty
        #
        # state = self._get_state()
        #
        # if self.enable_gui:
        #     self._render()
        #
        return state, reward, self.done, {}

    def _get_state(self):
        """Get current state"""
        # Sensor readings
        sensor_distances = [self._get_sensor_distance(angle) 
                           for angle in self.sensor_angles]
        
        # Target relative position and angle
        target_pos = self.tables[self.target_table_idx]
        relative_pos = target_pos - self.agent_pos
        distance_to_target = np.linalg.norm(relative_pos)
        angle_to_target = math.atan2(relative_pos[1], relative_pos[0]) - self.agent_angle
        angle_to_target = math.atan2(math.sin(angle_to_target), math.cos(angle_to_target))
        
        state = np.array([
            self.agent_pos[0] / self.width,      # Normalized position
            self.agent_pos[1] / self.height,
            math.cos(self.agent_angle),          # Orientation
            math.sin(self.agent_angle),
            distance_to_target / (self.width + self.height),  # Normalized distance
            math.cos(angle_to_target),           # Target angle
            math.sin(angle_to_target),
            *[d / self.max_sensor_range for d in sensor_distances]  # Normalized sensor readings
        ])
        
        return state.astype(np.float32)

    def _is_valid_position(self, pos):
        """Check if position is valid"""
        x, y = pos
        
        # Check boundaries
        if not (self.agent_radius <= x <= self.width - self.agent_radius and 
                self.agent_radius <= y <= self.height - self.agent_radius):
            return False
        
        # Check collision with tables
        for table in self.tables:
            if np.linalg.norm(pos - table) < self.table_radius + self.agent_radius:
                return False
        
        return True

    def _get_sensor_distance(self, angle_offset):
        """Get sensor distance reading"""
        angle = self.agent_angle + angle_offset
        
        for dist in np.linspace(0, self.max_sensor_range, num=50):
            probe = self.agent_pos + dist * np.array([math.cos(angle), math.sin(angle)])
            if not self._is_valid_position(probe):
                return dist
        
        return self.max_sensor_range

    def _render(self):
        """Render environment"""
        if not self.enable_gui:
            return
            
        self.window.fill((255, 255, 255))

        #This avoids the freezing of the visuals (not responding) window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        def to_px(pos):
            return (int(pos[0] * self.screen_scale), 
                   int(self.height * self.screen_scale - pos[1] * self.screen_scale))
        
        # Draw tables
        for i, table in enumerate(self.tables):
            color = (255, 0, 0) if i == self.target_table_idx else (139, 69, 19)
            pygame.draw.circle(self.window, color, to_px(table), 
                             int(self.table_radius * self.screen_scale))
        
        # Draw agent
        agent_px = to_px(self.agent_pos)
        pygame.draw.circle(self.window, (0, 102, 204), agent_px, 
                          int(self.agent_radius * self.screen_scale))
        
        # Draw orientation
        end_pos = self.agent_pos + 0.3 * np.array([math.cos(self.agent_angle), 
                                                   math.sin(self.agent_angle)])
        pygame.draw.line(self.window, (0, 0, 255), agent_px, to_px(end_pos), 3)
        
        # Draw sensors
        for angle_offset in self.sensor_angles:
            sensor_distance = self._get_sensor_distance(angle_offset)
            sensor_end = self.agent_pos + sensor_distance * np.array([
                math.cos(self.agent_angle + angle_offset),
                math.sin(self.agent_angle + angle_offset)
            ])
            pygame.draw.line(self.window, (150, 150, 150), agent_px, to_px(sensor_end), 1)
        
        pygame.display.flip()
        pygame.time.delay(50)

    def close(self):
        """Close environment"""
        if self.enable_gui:
            pygame.quit()

    def get_state_size(self):
        """Get state space size"""
        return len(self._get_state())

    def get_action_size(self):
        """Get action space size"""
        return self.action_space_size 