import torch
import typing
from typing import Dict, List
from .map_layouts import MAP_LAYOUTS, ACTIVE_MAPS_EVALUATION, ACTIVE_MAPS_TRAINING

from vmas.simulator.core import Agent, Landmark, World, Sphere, Box, Entity
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils
from vmas.simulator.dynamics.forward import Forward

import numpy as np

len_active_maps_training = len(ACTIVE_MAPS_TRAINING)
len_active_evaluation_maps = len(ACTIVE_MAPS_EVALUATION)


class PathPlanningScenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # 1. Configuration
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 360)  # Paper uses 360 rays
        self.lidar_range = kwargs.pop("lidar_range", 2.0)    # Paper uses 2m range
        self.training = kwargs.pop("training", True)
        self.debug = kwargs.pop("debug_prints", False)
        
        self.grid_size = kwargs.pop("grid_size", 0.2)
        self.map_cols = kwargs.pop("map_cols", 28)
        self.map_rows = kwargs.pop("map_rows", 18)
        self.map_choice = 0

        self.world_x_bound = self.map_cols * self.grid_size / 2.0  # 2.8m
        self.world_y_bound = self.map_rows * self.grid_size / 2.0  # 1.8m
        
        # L - Max initial distance between agent and goal (Curriculum setting)
        self.set_max_dist(kwargs.pop("max_dist", 2.0) )
        # Safety margin for spawning near walls/obstacles (Agent radius is 0.1m)
        self.safety_margin = kwargs.pop("safety_margin", 0.125) 

        # Set nominal dt for obstacle force scaling
        self.nominal_dt = kwargs.pop("nominal_dt", 0.1) 
        if self.debug:
            print(f"[DEBUG] Scenario nominal_dt set to: {self.nominal_dt}")

        # World Bounds (Paper uses approx 2-3m area based on obstacle size)
        world = World(batch_dim, 
                      device, 
                      dt=self.nominal_dt, 
                      drag=0.25)
        
        self.max_moving_obstacles = 0  
        self.max_static_walls = 0

        # Pre-parse maps
        self.parsed_maps = {}
        for map_id, layout in MAP_LAYOUTS.items():
            w_count = sum(row.count('W') for row in layout)
            m_count = sum(row.count('U') + row.count('S') for row in layout)
            self.max_static_walls = max(self.max_static_walls, w_count)
            self.max_moving_obstacles = max(self.max_moving_obstacles, m_count)
            
            # Pre-parse map to avoid overhead in reset
            self.parsed_maps[map_id] = self.parse_map_layout(map_id, grid_size=self.grid_size)

        # 2. The Robot (Agent)
        # Mecanum wheels -> omnidirectional (Holonomic). VMAS agents are holonomic by default.
        self.agent = Agent(
            name="robot",
            shape=Box(length=0.2, width=0.2),  # Paper: 20x20cm gray square
            color=Color.GRAY,
            u_range=1.0, # Revert: Set back to 1.0 for direct velocity control
            sensors=[Lidar(
                world,
                n_rays=self.n_lidar_rays,
                max_range=self.lidar_range,
                entity_filter=lambda e: e.collide,  # Fix: Only cast rays against collidable entities
                render=False,
            )],
            action_size=1, # 1 action dimension for discrete action indexing
            discrete_action_nvec=[8],  # 8 discrete actions (N, S, W, E, NW, NE, SW, SE
            dynamics=Forward()
        )
        world.add_agent(self.agent)

        # 3. The Goal
        self.goal = Landmark(
            name="goal",
            shape=Sphere(radius=0.05),
            color=Color.GREEN,
            collide=False
        )
        world.add_landmark(self.goal)
        self.prev_dist_to_goal = torch.zeros( batch_dim, dtype=torch.float32, device=device)

        # 4. Moving Obstacles (Distractors)
        # We create them as Agents so they can move, but we will script their motion.
        self.moving_obstacles = []
        for i in range(self.max_moving_obstacles):
            obstacle = Agent(
                            name=f"moving_obs_{i}",
                            shape=Box(length=0.2, width=0.2),  # Paper: 20x20cm white square
                            color=Color.WHITE,
                            collide=True,
                            mass=1000,  # Heavy so the robot doesn't push them easily
                            u_range=1e6,  # Fix: Increase u_range to allow high forces for heavy scripted agents
                            action_script=self.patrol_script,  # THIS IS KEY: They follow a script
            )
            # Store patrol metadata (start, end, speed) in the object for easy access
            obstacle.patrol_start = torch.zeros((batch_dim, 2), device=device)
            obstacle.patrol_end = torch.zeros((batch_dim, 2), device=device)
            obstacle.patrol_speed = 1.5  
            obstacle.nominal_p_gain = 2000.0  # Fix: Increase gain to move heavy mass
            world.add_agent(obstacle)
            self.moving_obstacles.append(obstacle)

        # 5. Static Walls
        self.walls = []
        for i in range(self.max_static_walls):
            wall = Landmark(
                name=f"wall_{i}",
                shape=Box(length=0.2, width=0.2),  # Default size, changed in reset
                color=Color.WHITE,
                collide=True
            )
            world.add_landmark(wall)
            self.walls.append(wall)

        self.obstacle_entities = self.moving_obstacles + self.walls
        
        # 6. Visual Border (Non-collidable)
        self.border_segments = []
        # We'll create 4 thin rectangles for the border with fixed dimensions
        border_thickness = 0.02
        
        # Horizontal borders (top/bottom) - length matches map width
        for i in range(2):
            border = Landmark(
                name=f"border_h_{i}",
                shape=Box(length=(self.map_cols * self.grid_size) + 0.2, width=border_thickness),
                color=Color.BLACK,
                collide=True,  # was False: make borders collidable so lidar can see them and they block the robot
            )
            # Note: we do NOT add these borders to self.obstacle_entities; they are separate "outer" boundaries
            world.add_landmark(border)
            self.border_segments.append(border)
        
        # Vertical borders (left/right) - length matches map height
        for i in range(2):
            border = Landmark(
                name=f"border_v_{i}",
                shape=Box(length=(self.map_rows * self.grid_size) + 0.2, width=border_thickness),
                color=Color.BLACK,
                collide=True,  # was False
            )
            world.add_landmark(border)
            self.border_segments.append(border)
        
        # Initialize global time for the batch
        self.global_time = torch.zeros(batch_dim, device=device)
        
        return world
    
    def parse_map_layout(self, map_id, grid_size=0.2):
        """
        Parses ASCII map into continuous coordinates.
        Returns: 
           walls: list of [x, y]
           moving_starts: list of dict {'pos': [x, y], 'type': char}
        """
        layout = MAP_LAYOUTS[map_id]
        rows = len(layout)
        cols = len(layout[0])
        
        # Calculate world dimensions based on grid
        width_m = cols * grid_size
        height_m = rows * grid_size
        
        # Offsets to center the map at (0,0)
        offset_x = width_m / 2.0
        offset_y = height_m / 2.0
        
        walls_pos = []
        moving_pos = []

        for r, row_str in enumerate(layout):
            for c, char in enumerate(row_str):
                # Convert Grid index -> Continuous World Coordinate
                # Formula: index * size - offset + half_size (to center in cell)
                cx = (c * grid_size) - offset_x + (grid_size / 2)
                # Invert Y because array index increases downwards, but World Y increases upwards
                cy = ((rows - 1 - r) * grid_size) - offset_y + (grid_size / 2)

                if char == 'W':
                    walls_pos.append([cx, cy])
                elif char in {'U', 'S'}:
                    moving_pos.append({'pos': [cx, cy], 'type': char})
        
        return walls_pos, moving_pos

    def patrol_script(self, agent: Agent, world: World):
        center = (agent.patrol_start + agent.patrol_end) / 2
        amplitude = (agent.patrol_end - agent.patrol_start) / 2
        
        # Fix: Check if patrol route is valid (non-zero amplitude)
        # If amplitude is near zero, this obstacle is not active on this map
        amplitude_norm = torch.linalg.norm(amplitude, dim=-1, keepdim=True)
        is_active = amplitude_norm > 1e-3
        
        # Use the tracked global time
        t = self.global_time.unsqueeze(-1) # Shape (batch_dim, 1)
        
        # Calculate the desired position based on sinusoidal movement
        # Phase shift -pi/2 ensures it starts at 'start' (sin(-pi/2) = -1)
        desired_pos = center + amplitude * torch.sin(t * agent.patrol_speed - 1.5708)
        
        # Calculate the position error
        error = desired_pos - agent.state.pos
        
        # Adjust the P-gain to be time-step independent.
        # Force is scaled by (Nominal DT / Actual DT) to maintain the same dynamic response
        # Nominal P-gain was tuned for dt=0.1
        dt_ratio = self.nominal_dt / world.dt
        effective_gain = agent.nominal_p_gain * dt_ratio
        
        # The output is the commanded force
        # Fix: Set force to zero for inactive obstacles
        force = error * effective_gain
        force = torch.where(is_active, force, torch.zeros_like(force))
        
        agent.action.u = force

    def _is_collision_free(self, pos, map_id):
        """Checks if a given position is too close to any obstacle in the map layout."""
        # Check against walls
        walls_data, moving_data = self.parsed_maps[map_id]
        
        pos_tensor = torch.tensor(pos, dtype=torch.float32)

        # Agent radius ~0.1m, ensure clearance >= radius + safety_margin
        agent_radius = 0.1
        effective_margin = agent_radius + self.safety_margin

        # Check against static walls
        for w_pos in walls_data:
            w_pos_tensor = torch.tensor(w_pos, dtype=torch.float32)
            dist = torch.linalg.norm(pos_tensor - w_pos_tensor)
            if dist < effective_margin:
                return False
        
        # Check against moving obstacle start positions AND their patrol paths
        for m_data in moving_data:
            m_pos_tensor = torch.tensor(m_data['pos'], dtype=torch.float32)
            
            # Check start position
            dist = torch.linalg.norm(pos_tensor - m_pos_tensor)
            if dist < effective_margin:
                return False
            
            # Check end position (calculate it the same way as in reset_world_at)
            p_end = m_pos_tensor.clone()
            otype = m_data['type']
            grid_size = self.grid_size
            
            if otype == 'U':
                p_end[1] += 5 * grid_size
            elif otype == 'S':
                p_end[0] += 5 * grid_size
            
            dist_end = torch.linalg.norm(pos_tensor - p_end)
            if dist_end < effective_margin:
                return False
            
            # Check along the patrol line (sample points between start and end)
            for alpha in [0.25, 0.5, 0.75]:
                interpolated_pos = m_pos_tensor + alpha * (p_end - m_pos_tensor)
                dist_path = torch.linalg.norm(pos_tensor - interpolated_pos)
                if dist_path < effective_margin:
                    return False
                
        # Also check against the outer map boundary
        if abs(pos[0]) > self.world_x_bound or abs(pos[1]) > self.world_y_bound:
             return False
             
        return True


    def reset_world_at(self, env_index: int = None):
        # 1. Decide which Maps to use (Mixed Batch Logic)
        # If env_index is None when initializing at the start of training or evaluation(reset all),
        # we pick randoms for everyone

        active_training_maps_tensor = torch.tensor(ACTIVE_MAPS_TRAINING, device=self.world.device)
        active_evaluation_maps_tensor = torch.tensor(ACTIVE_MAPS_EVALUATION, device=self.world.device)

        if env_index is None:
            # Resetting ALL envs.
            indice = None  # If set to None, VMAS applies to all envs
            # Reset time for all
            self.global_time.fill_(0.0)

            if self.training:
                # Randomly assign Map 0 or Map 1 to each env in the batch
                map_choices_idx = torch.randint(0,
                                                len_active_maps_training,
                                                (self.world.batch_dim,),
                                                device=self.world.device
                                                )
                map_choices = active_training_maps_tensor[map_choices_idx]
            else:
                # Evaluation Mode: Force Map 2 (High Level)
                map_choices_idx = torch.randint(0,
                                                len_active_evaluation_maps,
                                                (self.world.batch_dim,),
                                                device=self.world.device
                                                )
                map_choices = active_evaluation_maps_tensor[map_choices_idx]
        else:
            # Resetting SINGLE env
            indice = env_index
            # Reset time for this env
            self.global_time[env_index] = 0.0

            if self.training:
                map_choices_idx = torch.randint(0,
                                                len_active_maps_training,
                                                (1,),
                                                device=self.world.device
                                                )
                map_choices = active_training_maps_tensor[map_choices_idx]
                self.map_choice = map_choices.item()  # TODO: does anything???
            else:
                map_choices_idx = torch.randint(0,
                                                len_active_evaluation_maps,
                                                (1,),
                                                device=self.world.device
                                                )
                map_choices = active_evaluation_maps_tensor[map_choices_idx]
                self.map_choice = map_choices.item()

        # 2. Configure Maps (Iterate via CPU to handle mixed maps)
        # First, reset all entities to "Out of Bounds" for these indices
        out_of_bounds = torch.tensor([self.world_x_bound + 1.0, self.world_y_bound + 1.0], device=self.world.device)
        for entity in self.obstacle_entities:
            entity.set_pos(out_of_bounds, batch_index=indice)
            if hasattr(entity, 'state') and entity.state.vel is not None:
                zero_vel = torch.zeros_like(entity.state.vel[0])
                entity.set_vel(zero_vel, batch_index=indice)
            if hasattr(entity, 'state') and entity.state.rot is not None:
                zero_rot = torch.zeros_like(entity.state.rot[0])
                entity.set_rot(zero_rot, batch_index=indice)

        # Apply specific map configurations per environment
        cpu_map_choices = map_choices.tolist()
        if env_index is None:
            cpu_indices = list(range(self.world.batch_dim))
        else:
            # Only configure the requested env index
            cpu_indices = [env_index]

        for i, map_id in zip(cpu_indices, cpu_map_choices):
            walls_data, moving_data = self.parsed_maps[map_id]
            
            # Half robot length (robot Box length is 0.2 -> half is 0.1)
            half_robot_len = 0.1

            # Set Visual Border (4 sides of the map), moved outward by half a robot length
            # Top border
            self.border_segments[0].set_pos(
                torch.tensor([0.0, self.world_y_bound + half_robot_len], device=self.world.device), 
                batch_index=i
            )
            self.border_segments[0].set_rot(torch.tensor([0.0], device=self.world.device), batch_index=i)
            
            # Bottom border
            self.border_segments[1].set_pos(
                torch.tensor([0.0, -self.world_y_bound - half_robot_len], device=self.world.device), 
                batch_index=i
            )
            self.border_segments[1].set_rot(torch.tensor([0.0], device=self.world.device), batch_index=i)
            
            # Left border
            self.border_segments[2].set_pos(
                torch.tensor([-self.world_x_bound - half_robot_len, 0.0], device=self.world.device), 
                batch_index=i
            )
            self.border_segments[2].set_rot(torch.tensor([1.5708], device=self.world.device), batch_index=i)
            
            # Right border
            self.border_segments[3].set_pos(
                torch.tensor([self.world_x_bound + half_robot_len, 0.0], device=self.world.device), 
                batch_index=i
            )
            self.border_segments[3].set_rot(torch.tensor([1.5708], device=self.world.device), batch_index=i)
            
            # Set Walls
            for w_idx, w_pos in enumerate(walls_data):
                if w_idx < len(self.walls):
                    self.walls[w_idx].set_pos(torch.tensor(w_pos, device=self.world.device), batch_index=i)
            
            # Set Moving Obstacles
            for m_idx, m_data in enumerate(moving_data):
                if m_idx < len(self.moving_obstacles):
                    obs = self.moving_obstacles[m_idx]
                    start_pos = torch.tensor(m_data['pos'], device=self.world.device)
                    obs.set_pos(start_pos, batch_index=i)
                    
                    # Randomly decide Patrol Type 
                    p_start = start_pos.clone()
                    p_end = start_pos.clone()
                    
                    otype = m_data['type']
                    if otype == 'U':
                        # Up/Down. Starts at map pos, goes down 4 cells.
                        p_end[1] += 5 * self.grid_size
                    elif otype == 'S':
                        # Sideways. Starts at map pos, goes right 4 cells.
                        p_end[0] += 5 * self.grid_size

                    obs.patrol_start[i] = p_start
                    obs.patrol_end[i] = p_end
            
            # 3. Reset Robot and Goal with Constraints (Goal-First Logic)
            placed = False
            attempts = 0

            # Define bounds for random sampling
            x_range = [-self.world_x_bound, self.world_x_bound]
            y_range = [-self.world_y_bound, self.world_y_bound]
            
            while not placed and attempts < 1000:
                # 3a. Randomly sample Goal Position in the 5.6m x 3.6m box
                goal_pos_np = np.array([
                    np.random.uniform(x_range[0], x_range[1]),
                    np.random.uniform(y_range[0], y_range[1])
                ])
                
                # Check if Goal position is collision-free
                if not self._is_collision_free(goal_pos_np, map_id):
                    attempts += 1
                    continue
                
                # 3b. Randomly sample Agent Position (within max_dist of goal)
                # 1. Sample direction (angle)
                angle = np.random.uniform(0, 2 * np.pi)
                # 2. Sample distance (0.11m min to prevent trivial rewards and allow
                # at least one movement to reach goal, up to L)
                distance = np.random.uniform(0.13, self._max_dist)
                
                # Calculate agent position relative to goal
                agent_pos_candidate = goal_pos_np + distance * np.array([np.cos(angle), np.sin(angle)])
                
                # Clamp agent position to the 5.6m x 3.6m boundary 
                agent_pos_np = np.array([
                    np.clip(agent_pos_candidate[0], x_range[0], x_range[1]),
                    np.clip(agent_pos_candidate[1], y_range[0], y_range[1])
                ])

                # Recalculate distance after clamping to ensure it is still valid
                final_dist = np.linalg.norm(agent_pos_np - goal_pos_np)
                
                # Check constraints: 
                # 1. Agent must be collision free (checked inside _is_collision_free, boundary check needed here too)
                # 2. Agent must be within max_dist (L) of the goal after clamping.
                # 3. Agent cannot be trivially close to the goal (min dist 0.2m)
                
                # Note: The boundary check is now redundant if we sample in the bounds, 
                # but we must re-check collision and the distance constraint.
                if not self._is_collision_free(agent_pos_np, map_id) or \
                   final_dist > self._max_dist or \
                   final_dist < 0.13:
                    attempts += 1
                    continue

                # Placement successful
                self.agent.set_pos(torch.tensor(agent_pos_np, device=self.world.device), batch_index=i)
                self.goal.set_pos(torch.tensor(goal_pos_np, device=self.world.device), batch_index=i)

                new_initial_dist = torch.linalg.norm(self.agent.state.pos[i] - self.goal.state.pos[i])
                self.prev_dist_to_goal[i] = new_initial_dist.clone().detach()
                placed = True

            if not placed:
                # Fallback in case of failure (should be rare with 1000 attempts)
                print(f"Warning: Failed to place agent/goal after 1000 attempts in environment {i}. Placing randomly.")
                # Fallback positions must also respect the 5.6m x 3.6m bounds
                self.agent.set_pos(torch.tensor([np.random.uniform(x_range[0], x_range[1]),
                                                 np.random.uniform(y_range[0], y_range[1])],
                                                device=self.world.device).unsqueeze(0), batch_index=i)
                self.goal.set_pos(torch.tensor([np.random.uniform(x_range[0], x_range[1]),
                                                np.random.uniform(y_range[0], y_range[1])],
                                               device=self.world.device).unsqueeze(0), batch_index=i)


    def process_action(self, agent: Agent):
        # Increment time once per step (triggered by the first agent, the robot)
        if agent == self.agent:
            self.global_time += self.world.dt

        if agent.name.startswith("moving_obs"):
            # Fix: VMAS automatically calls action_script before this method.
            # We just apply the action (force) that was set in the script.
            agent.state.force[:] = agent.action.u
            return

        if agent.name == "robot":
            # Robot speed is calculated to ensure a fixed 0.1m displacement per step.
            # Speed = (0.1m) / dt. Current dt=0.1, so speed=1.0.
            speed = 0.1 / self.world.dt
            diag_speed = speed * 0.707

            u = agent.action.u[:, 0]      # (B,), values in [-1, 1]
            n = 8 #Number of discrete actions
            #This needs to happen because VMAS does some weird 
            # continuous mapping and provides the action as a continuous value in [-1, 1]
            action_index = torch.round((u + 1) / 2 * (n - 1)).long().clamp(0, n-1)
            
            vx = torch.zeros_like(action_index, dtype=torch.float32)
            vy = torch.zeros_like(action_index, dtype=torch.float32)
            rot = torch.zeros_like(action_index, dtype=torch.float32)

            # 0: Up, 1: Down, 2: Left, 3: Right
            mask_0 = (action_index == 0)
            vy[mask_0] = speed
            rot[mask_0] = 1.57
            
            mask_1 = (action_index == 1)
            vy[mask_1] = -speed
            rot[mask_1] = -1.57
            
            mask_2 = (action_index == 2)
            vx[mask_2] = -speed
            rot[mask_2] = 3.14
            
            mask_3 = (action_index == 3)
            vx[mask_3] = speed
            rot[mask_3] = 0.0 
            
            # Diagonals
            mask_4 = (action_index == 4) # FL
            vx[mask_4] = -diag_speed; vy[mask_4] = diag_speed; rot[mask_4] = 2.35
            
            mask_5 = (action_index == 5) # FR
            vx[mask_5] = diag_speed; vy[mask_5] = diag_speed; rot[mask_5] = 0.78
            
            mask_6 = (action_index == 6) # BL
            vx[mask_6] = -diag_speed; vy[mask_6] = -diag_speed; rot[mask_6] = -2.35
            
            mask_7 = (action_index == 7) # BR
            vx[mask_7] = diag_speed; vy[mask_7] = -diag_speed; rot[mask_7] = -0.78

            # Revert to Direct Velocity Control (Kinematic)
            agent.state.vel[:, 0] = vx
            agent.state.vel[:, 1] = vy
            agent.state.rot[:, 0] = rot
            # Zero out forces so physics doesn't add extra acceleration
            agent.state.force[:] = 0.0

    def reward(self, agent: Agent):
        # Paper Reward:
        # +1 if reach goal
        # -1 if collision
        # -0.01 time penalty
        
        # Distance to goal
        dist = torch.linalg.norm(agent.state.pos - self.goal.state.pos, dim=-1)
        is_at_goal = dist < 0.10

        progress = (self.prev_dist_to_goal - dist)
        decreased = progress >= 0
        increased = progress < 0

        #Lidar penalty
        MIN_DIST_THRESHOLD = 0.25 # 1.0m de distance réelle (0.5 * 2.0m)
        PROXIMITY_PENALTY_SCALE = 0.5 # Ajustez ce facteur

        lidar_distances = agent.sensors[0].measure()
        min_dist = torch.min(lidar_distances, dim=-1).values
        proximity_penalty = torch.zeros_like(min_dist)
        too_close_mask = min_dist < MIN_DIST_THRESHOLD
        penalty_magnitude = MIN_DIST_THRESHOLD - min_dist[too_close_mask]
        proximity_penalty[too_close_mask] = -PROXIMITY_PENALTY_SCALE * penalty_magnitude



        # Collision Check
        is_collision = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)
        for entity in self.world.agents + self.world.landmarks:
            if entity != agent and entity.collide:
                 is_collision = is_collision | self.world.is_overlapping(agent, entity)

        rews = torch.zeros_like(dist)
        rews[is_at_goal] += 5.0
        rews[is_collision] -= 10.0
        rews[decreased] += 0.5
        rews[increased] -= 0.5
        rews += -0.01 # Time penalty
        rews += proximity_penalty

        self.prev_dist_to_goal = dist.clone().detach()        
        return rews

    def done(self) -> torch.Tensor:
        """Episode termination: reached goal, collision, or time-limit truncation.

        NOTE on VMAS `terminated_truncated`:
        - Environment always calls this method to compute **terminated**.
        - If `make_env(..., max_steps=..., terminated_truncated=True)` is used,
          VMAS will also compute a separate **truncated** flag from `max_steps`. we do not have to take care of truncation
        """
        # Distance to goal
        dist = torch.linalg.norm(self.agent.state.pos - self.goal.state.pos, dim=-1)
        reached_goal = dist < 0.10  # same threshold as in reward

        # Collision with any collidable entity
        is_collision = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)
        for entity in self.world.agents + self.world.landmarks:
            if entity != self.agent and entity.collide:
                is_collision |= self.world.is_overlapping(self.agent, entity)

        # Terminate if goal reached or collision
        terminated = reached_goal | is_collision
        return terminated

    def observation(self, agent: 'Agent'):
        batch_dim = self.world.batch_dim
        device = self.world.device

        # --- 1. Lidar Mesures (360 éléments) ---
        lidar_distances = agent.sensors[0].measure()/self.lidar_range # (B, 360)

        # Création des angles (0-2*pi)
        angle_vector = torch.linspace(start=0, end=2*np.pi, steps=self.n_lidar_rays, device=device).unsqueeze(0)
        lidar_angles = angle_vector.repeat(batch_dim, 1) # (B, 360)
        # Empilement et Transposition: (B, 360, 2) -> (B, 2, 360)
        lidar_mesures = torch.stack((lidar_distances, lidar_angles), dim=2).transpose(1, 2)
        # Total éléments: B * 720. Nous utilisons B = batch_dim.
        lidar_input_reshaped = lidar_mesures.reshape(batch_dim, 2, 18, 20)
        
        # --- 2. Cible Locale (80 éléments) ---
        delta_pos = self.goal.state.pos - agent.state.pos # (B, 2)
        
        # --- MODIFICATION: Passage aux Coordonnées Polaires ---

        # 1. Calcul de la Distance (r)
        # torch.norm(..., dim=1) calcule la norme sur la dimension (X, Y)
        distance = torch.norm(delta_pos, dim=1, keepdim=True) # (B, 1)

        # 2. Calcul de l'Angle (theta)
        # torch.atan2(Y, X) donne l'angle dans [-pi, pi]
        angle = torch.atan2(delta_pos[:, 1], delta_pos[:, 0]).unsqueeze(1) # (B, 1)
        agent_orientation = agent.state.rot
        angle = angle - agent_orientation
        
        # L'angle peut être normalisé si nécessaire, par exemple en divisant par pi 
        # angle = angle / np.pi 

        # Concaténation des coordonnées polaires (Distance, Angle)
        polar_coords = torch.cat((distance, angle), dim=1) # (B, 2)
        
        # 40 copies de (Distance, Angle) -> 80 éléments (B, 80)
        target_vector = polar_coords.repeat(1, 40)
        target_input_reshaped = target_vector.reshape(batch_dim, 2, 2, 20)

        # --- 3. Concaténation Finale (B, 2, 20, 20) ---
        # Concaténation sur dim=2 (Hauteur: 18 + 2 = 20)
        raw_input_800 = torch.cat((lidar_input_reshaped, target_input_reshaped), dim=2 )
        return raw_input_800

    def info(self, agent: Agent) -> Dict[str, torch.Tensor]:
        """
        Per-agent info dict.

        Returns the current position of the 'robot' agent for all parallel envs.
        Shape: (batch_dim, 2).
        """
        # Always report robot position, regardless of which agent 'info' is called for
        if agent.name == self.agent.name:
            return self.agent.state.pos.clone()

    def set_max_dist(self, max_dist: float):
        """Sets the maximum spawning distance for curriculum learning."""
        if max_dist < 0.13:
            raise ValueError("max_dist must be at least 0.13 to allow valid spawning.")
        self._max_dist = max_dist
        if self.debug:
            print(f"[Scenario] Set spawn max_dist to: {self._max_dist}")

    @property
    def max_dist(self) -> float:
        """Gets the current maximum spawning distance."""
        return self._max_dist
