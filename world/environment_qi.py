"""
Restaurant Delivery Robot RL Environment for SAC Training
Created by Qi for 2AMC15 Assignment 2

This environment implements a complete reinforcement learning task where
a robot must navigate a restaurant to pick up and deliver orders to tables.

Key Features:
- Continuous state space (10D vector)
- Discrete action space (5 actions mapped to continuous control)
- Delivery task with pickup/delivery mechanics
- Reward shaping for efficient learning
- Episode management with success/failure conditions
- Real-world restaurant scenario motivation
"""

import numpy as np
import math
import random
import pygame
from pygame import gfxdraw
from pathlib import Path
from world.continuous_space import ContinuousSpace
from typing import Tuple, Dict, Any, Optional

# Visual settings
WINDOW_SIZE = (800, 600)
TABLE_COLOR = (139, 69, 19)
AGENT_COLOR = (0, 102, 204)
AGENT_WITH_ORDER_COLOR = (255, 165, 0)  # Orange when carrying order
BACKGROUND_COLOR = (255, 255, 255)
SENSOR_COLOR = (150, 150, 150)
PICKUP_COLOR = (0, 255, 0)  # Green for pickup point
TARGET_COLOR = (255, 0, 0)  # Red for target table
COMPLETED_COLOR = (128, 128, 128)  # Gray for completed deliveries

class RestaurantDeliveryEnvironment:
    """
    Restaurant Delivery Robot RL Environment
    
    **Business Case**: A restaurant wants to automate food delivery to tables
    using an autonomous robot. The robot must efficiently navigate between
    the kitchen (pickup point) and customer tables while avoiding obstacles.
    
    **Task**: Pick up orders from kitchen and deliver to designated tables
    **Success Metric**: Complete deliveries while minimizing time and collisions
    **Real-world Impact**: Reduces labor costs, improves service consistency
    """
    
    def __init__(self, 
                 space_file: Optional[Path] = None,
                 width: float = 10.0, 
                 height: float = 10.0,
                 table_radius: float = 0.5,
                 n_tables: int = 3,
                 max_episode_steps: int = 500,
                 pickup_radius: float = 0.3,
                 delivery_radius: float = 0.4,
                 agent_radius: float = 0.2,
                 enable_gui: bool = True,
                 seed: int = 42):
        
        # Load restaurant layout
        if space_file is not None and space_file.exists():
            space = ContinuousSpace.load(space_file)
            self.width = space.width
            self.height = space.height
            self.tables = space.tables
            self.table_radius = space.table_radius
        else:
            self.width = width
            self.height = height
            self.table_radius = table_radius
            self.tables = self._generate_tables(n_tables, seed)
        
        # Environment parameters
        self.max_episode_steps = max_episode_steps
        self.pickup_radius = pickup_radius
        self.delivery_radius = delivery_radius
        self.agent_radius = agent_radius
        self.enable_gui = enable_gui
        
        # Agent physical properties
        self.max_velocity = 0.5
        self.max_rotation = 0.5
        self.sensor_angles = [0, -np.radians(30), np.radians(30)]  # Front, left, right
        self.max_sensor_range = 5.0
        
        # Task state
        self.pickup_point = np.array([1.0, 1.0])  # Kitchen location
        self.has_order = False
        self.current_target_table = None
        self.completed_deliveries = 0
        self.total_orders = 3  # Number of orders to complete per episode
        
        # Episode tracking
        self.step_count = 0
        self.episode_reward = 0.0
        self.collision_count = 0
        
        # Rendering
        self.window = None
        self.screen_scale = 50
        
        if self.enable_gui:
            pygame.init()
            self.window = pygame.display.set_mode(WINDOW_SIZE)
            pygame.display.set_caption("Restaurant Delivery Robot - SAC Training")
        
        # Initialize environment
        self.reset()
    
    def _generate_tables(self, n_tables: int, seed: int) -> list:
        """Generate random table positions"""
        random.seed(seed)
        np.random.seed(seed)
        tables = []
        
        for _ in range(n_tables):
            # Ensure tables are not too close to pickup point or boundaries
            while True:
                x = np.random.uniform(self.table_radius + 1.0, self.width - self.table_radius - 1.0)
                y = np.random.uniform(self.table_radius + 1.0, self.height - self.table_radius - 1.0)
                pos = np.array([x, y])
                
                # Check distance from pickup point
                if np.linalg.norm(pos - self.pickup_point) > 2.0:
                    # Check distance from other tables
                    if all(np.linalg.norm(pos - table) > 2.0 * self.table_radius for table in tables):
                        tables.append(pos)
                        break
        
        return tables
    
    def reset(self) -> Dict[str, Any]:
        """Reset environment for new episode"""
        # Reset agent state
        self.agent_pos = np.array([0.5, 0.5])  # Start near pickup point
        self.agent_angle = 0.0
        
        # Reset task state
        self.has_order = False
        self.current_target_table = None
        self.completed_deliveries = 0
        
        # Reset episode tracking
        self.step_count = 0
        self.episode_reward = 0.0
        self.collision_count = 0
        
        # Assign first target table randomly
        self._assign_new_target()
        
        return self._get_observation()
    
    def step(self, action: Tuple[float, float]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one environment step
        
        Args:
            action: (velocity, rotation) tuple
            
        Returns:
            observation: Current state
            reward: Immediate reward
            done: Episode termination flag
            info: Additional information
        """
        self.step_count += 1
        
        # Clip actions to valid ranges
        velocity = np.clip(action[0], -self.max_velocity, self.max_velocity)
        rotation = np.clip(action[1], -self.max_rotation, self.max_rotation)
        
        # Update agent state
        self.agent_angle += rotation
        self.agent_angle = self.agent_angle % (2 * np.pi)  # Normalize angle
        
        # Calculate new position
        dx = velocity * math.cos(self.agent_angle)
        dy = velocity * math.sin(self.agent_angle)
        new_pos = self.agent_pos + np.array([dx, dy])
        
        # Check collision and update position
        collision = False
        if self._is_valid_position(new_pos):
            self.agent_pos = new_pos
        else:
            collision = True
            self.collision_count += 1
        
        # Calculate reward
        reward = self._calculate_reward(velocity, rotation, collision)
        self.episode_reward += reward
        
        # Check task completion
        self._check_task_completion()
        
        # Check episode termination
        done = self._is_episode_done()
        
        # Get observation
        obs = self._get_observation()
        
        # Render if enabled
        if self.enable_gui:
            self._render()
        
        # Info dictionary
        info = {
            'completed_deliveries': self.completed_deliveries,
            'has_order': self.has_order,
            'collision_count': self.collision_count,
            'step_count': self.step_count,
            'episode_reward': self.episode_reward,
            'success_rate': self.completed_deliveries / self.total_orders if self.total_orders > 0 else 0.0
        }
        
        return obs, reward, done, info
    
    def _calculate_reward(self, velocity: float, rotation: float, collision: bool) -> float:
        """
        Reward function designed for efficient delivery behavior
        
        Reward Components:
        1. Task completion: +5.0 for successful delivery, +1.0 for pickup
        2. Movement efficiency: Small negative for each step to encourage speed
        3. Collision penalty: -5.0 for hitting obstacles
        4. Progress reward: Positive for moving toward current objective
        """
        reward = 0.0
        
        # Step cost to encourage efficiency
        reward -= 0.01
        
        # Collision penalty
        if collision:
            reward -= 5.0
        
        # Progress toward current objective
        if self.has_order and self.current_target_table is not None:
            # Moving toward delivery target
            distance_to_target = np.linalg.norm(self.agent_pos - self.current_target_table)
            if distance_to_target < 1.0:  # Close to target
                reward += 0.1 * (1.0 - distance_to_target)  # Higher reward when closer
        elif not self.has_order:
            # Moving toward pickup point
            distance_to_pickup = np.linalg.norm(self.agent_pos - self.pickup_point)
            if distance_to_pickup < 1.0:  # Close to pickup
                reward += 0.05 * (1.0 - distance_to_pickup)
        
        # Movement reward (encourage forward motion, discourage excessive rotation)
        reward += 0.01 * abs(velocity) - 0.005 * abs(rotation)
        
        return reward
    
    def _check_task_completion(self):
        """Check if agent completed pickup or delivery"""
        agent_pos = self.agent_pos
        
        if not self.has_order:
            # Check pickup
            distance_to_pickup = np.linalg.norm(agent_pos - self.pickup_point)
            if distance_to_pickup <= self.pickup_radius:
                self.has_order = True
                self.episode_reward += 1.0  # Pickup reward
                print(f"📦 Order picked up! Delivery target: Table {self._get_table_id(self.current_target_table)}")
        
        elif self.current_target_table is not None:
            # Check delivery
            distance_to_target = np.linalg.norm(agent_pos - self.current_target_table)
            if distance_to_target <= self.delivery_radius:
                self.has_order = False
                self.completed_deliveries += 1
                self.episode_reward += 5.0  # Delivery reward
                print(f"🎯 Delivery completed! ({self.completed_deliveries}/{self.total_orders})")
                
                # Assign new target if more orders remain
                if self.completed_deliveries < self.total_orders:
                    self._assign_new_target()
                else:
                    self.current_target_table = None
    
    def _assign_new_target(self):
        """Assign a new random target table"""
        if len(self.tables) > 0:
            self.current_target_table = random.choice(self.tables)
    
    def _get_table_id(self, table_pos) -> int:
        """Get table ID for display purposes"""
        for i, table in enumerate(self.tables):
            if np.allclose(table, table_pos, atol=0.1):
                return i + 1
        return 0
    
    def _is_episode_done(self) -> bool:
        """Check if episode should terminate"""
        # Success: All deliveries completed
        if self.completed_deliveries >= self.total_orders:
            return True
        
        # Timeout: Maximum steps reached
        if self.step_count >= self.max_episode_steps:
            return True
        
        # Failure: Too many collisions
        if self.collision_count >= 10:
            return True
        
        return False
    
    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if position is valid (no collisions)"""
        x, y = pos
        
        # Check boundaries
        if not (self.agent_radius <= x <= self.width - self.agent_radius and 
                self.agent_radius <= y <= self.height - self.agent_radius):
            return False
        
        # Check table collisions
        for table in self.tables:
            distance = np.linalg.norm(pos - table)
            if distance < (self.table_radius + self.agent_radius):
                return False
        
        return True
    
    def _get_sensor_distance(self, angle_offset: float) -> float:
        """Get distance reading from sensor at given angle offset"""
        angle = self.agent_angle + angle_offset
        
        for dist in np.linspace(0, self.max_sensor_range, num=100):
            probe_pos = self.agent_pos + dist * np.array([math.cos(angle), math.sin(angle)])
            if not self._is_valid_position(probe_pos):
                return dist
        
        return float(self.max_sensor_range)
    
    def _get_observation(self) -> Dict[str, Any]:
        """
        Get current observation
        
        Returns comprehensive state information for SAC agent:
        - Agent position and orientation
        - Sensor distances for obstacle avoidance
        - Task state (has order, target locations)
        - Spatial relationships for navigation
        """
        # Sensor readings
        sensor_distances = [self._get_sensor_distance(angle) for angle in self.sensor_angles]
        
        # Task information
        pickup_point = self.pickup_point.copy()
        target_tables = [table.copy() for table in self.tables]
        
        observation = {
            "agent_pos": self.agent_pos.copy(),
            "agent_angle": self.agent_angle,
            "sensor_distances": sensor_distances,
            "has_order": self.has_order,
            "pickup_point": pickup_point,
            "current_target_table": self.current_target_table.copy() if self.current_target_table is not None else None,
            "target_tables": target_tables,
            "completed_deliveries": self.completed_deliveries,
            "total_orders": self.total_orders
        }
        
        return observation
    
    def _render(self):
        """Render the environment"""
        if not self.enable_gui or self.window is None:
            return
        
        self.window.fill(BACKGROUND_COLOR)
        
        def to_px(pos):
            return (int(pos[0] * self.screen_scale), 
                   int(self.height * self.screen_scale - pos[1] * self.screen_scale))
        
        # Draw tables
        for i, table in enumerate(self.tables):
            color = TARGET_COLOR if (self.current_target_table is not None and 
                                   np.allclose(table, self.current_target_table)) else TABLE_COLOR
            pygame.draw.circle(self.window, color, to_px(table), 
                             int(self.table_radius * self.screen_scale))
            
            # Table number
            font = pygame.font.Font(None, 24)
            text = font.render(str(i + 1), True, (255, 255, 255))
            text_rect = text.get_rect(center=to_px(table))
            self.window.blit(text, text_rect)
        
        # Draw pickup point
        pygame.draw.circle(self.window, PICKUP_COLOR, to_px(self.pickup_point), 
                          int(self.pickup_radius * self.screen_scale))
        font = pygame.font.Font(None, 20)
        text = font.render("KITCHEN", True, (0, 0, 0))
        text_rect = text.get_rect(center=to_px(self.pickup_point))
        self.window.blit(text, text_rect)
        
        # Draw agent
        agent_color = AGENT_WITH_ORDER_COLOR if self.has_order else AGENT_COLOR
        pygame.draw.circle(self.window, agent_color, to_px(self.agent_pos), 
                          int(self.agent_radius * self.screen_scale * 2))
        
        # Draw agent direction
        direction_end = self.agent_pos + 0.3 * np.array([math.cos(self.agent_angle), 
                                                         math.sin(self.agent_angle)])
        pygame.draw.line(self.window, (0, 0, 0), to_px(self.agent_pos), 
                        to_px(direction_end), 3)
        
        # Draw sensors
        for i, angle in enumerate(self.sensor_angles):
            sensor_distance = self._get_sensor_distance(angle)
            sensor_end = self.agent_pos + sensor_distance * np.array([
                math.cos(self.agent_angle + angle),
                math.sin(self.agent_angle + angle)
            ])
            pygame.draw.line(self.window, SENSOR_COLOR, to_px(self.agent_pos), 
                           to_px(sensor_end), 1)
        
        # Draw status information
        font = pygame.font.Font(None, 36)
        status_text = f"Deliveries: {self.completed_deliveries}/{self.total_orders}"
        if self.has_order:
            status_text += " | Carrying Order"
        else:
            status_text += " | Need Pickup"
        
        text = font.render(status_text, True, (0, 0, 0))
        self.window.blit(text, (10, 10))
        
        # Draw step count and reward
        info_text = f"Steps: {self.step_count} | Reward: {self.episode_reward:.2f} | Collisions: {self.collision_count}"
        text = font.render(info_text, True, (0, 0, 0))
        self.window.blit(text, (10, 50))
        
        pygame.display.flip()
        pygame.time.delay(50)
    
    def close(self):
        """Clean up resources"""
        if self.enable_gui and self.window is not None:
            pygame.quit()
    
    def get_success_rate(self) -> float:
        """Get current success rate"""
        return self.completed_deliveries / self.total_orders if self.total_orders > 0 else 0.0
    
    def get_episode_info(self) -> Dict[str, Any]:
        """Get comprehensive episode information"""
        return {
            'completed_deliveries': self.completed_deliveries,
            'total_orders': self.total_orders,
            'success_rate': self.get_success_rate(),
            'episode_reward': self.episode_reward,
            'step_count': self.step_count,
            'collision_count': self.collision_count,
            'efficiency': self.completed_deliveries / max(self.step_count, 1) * 100,
            'avg_reward_per_step': self.episode_reward / max(self.step_count, 1)
        } 