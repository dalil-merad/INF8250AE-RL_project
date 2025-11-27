import torch
import typing
from typing import Dict, List
from environment.map_layouts import MAP_LAYOUTS

from vmas.simulator.core import Agent, Landmark, World, Sphere, Box, Entity
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils

import numpy as np



class PathPlanningScenario(BaseScenario):
    def make_world(self, batch_dim: int , device: torch.device, **kwargs):
        # 1. Configuration
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 360) # Paper uses 360 rays
        self.lidar_range = kwargs.pop("lidar_range", 2.0)   # Paper uses 2m range
        self.training = kwargs.pop("training", True)
        self.debug = kwargs.pop("debug_prints", False)
        
        self.grid_size = kwargs.pop("grid_size", 0.2)
        self.map_cols = kwargs.pop("map_cols", 28)
        self.map_rows = kwargs.pop("map_rows", 18)

        self.world_x_bound = self.map_cols * self.grid_size / 2.0  # 2.8m
        self.world_y_bound = self.map_rows * self.grid_size / 2.0  # 1.8m
        
        # L - Max initial distance between agent and goal (Curriculum setting)
        self.set_max_dist(kwargs.pop("max_dist", 2.0) )
        # Safety margin for spawning near walls/obstacles (Agent radius is 0.1m)
        self.safety_margin = kwargs.pop("safety_margin", 0.25) 

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
            shape=Box(length=0.2, width=0.2), # Paper: 20x20cm gray square
            color=Color.GRAY,
            u_range=1.0, # Revert: Set back to 1.0 for direct velocity control
            sensors=[Lidar(
                world,
                n_rays=self.n_lidar_rays,
                max_range=self.lidar_range,
                entity_filter=lambda e: e.collide, # Fix: Only cast rays against collidable entities
                render=False
            )]
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

        # 4. Moving Obstacles (Distractors)
        # We create them as Agents so they can move, but we will script their motion.
        self.moving_obstacles = []
        for i in range(self.max_moving_obstacles):
            obstacle = Agent(
                            name=f"moving_obs_{i}",
                            shape=Box(length=0.2, width=0.2), # Paper: 20x20cm white square
                            color=Color.WHITE,
                            collide=True,
                            mass=1000, # Heavy so the robot doesn't push them easily
                            u_range=1e6, # Fix: Increase u_range to allow high forces for heavy scripted agents
                            action_script=self.patrol_script # THIS IS KEY: They follow a script
            )
            # Store patrol metadata (start, end, speed) in the object for easy access
            obstacle.patrol_start = torch.zeros((batch_dim, 2), device=device)
            obstacle.patrol_end = torch.zeros((batch_dim, 2), device=device)
            obstacle.patrol_speed = 1.5  
            obstacle.nominal_p_gain = 2000.0 # Fix: Increase gain to move heavy mass
            world.add_agent(obstacle)
            self.moving_obstacles.append(obstacle)

        # 5. Static Walls
        self.walls = []
        for i in range(self.max_static_walls):
            wall = Landmark(
                name=f"wall_{i}",
                shape=Box(length=0.2, width=0.2), # Default size, changed in reset
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
                shape=Box(length=self.map_cols * self.grid_size, width=border_thickness),
                color=Color.BLACK,
                collide=False
            )
            world.add_landmark(border)
            self.border_segments.append(border)
        
        # Vertical borders (left/right) - length matches map height
        for i in range(2):
            border = Landmark(
                name=f"border_v_{i}",
                shape=Box(length=self.map_rows * self.grid_size, width=border_thickness),
                color=Color.BLACK,
                collide=False
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

        # Check against static walls
        for w_pos in walls_data:
            w_pos_tensor = torch.tensor(w_pos, dtype=torch.float32)
            dist = torch.linalg.norm(pos_tensor - w_pos_tensor)
            if dist < self.safety_margin:
                return False
        
        # Check against moving obstacle start positions AND their patrol paths
        for m_data in moving_data:
            m_pos_tensor = torch.tensor(m_data['pos'], dtype=torch.float32)
            
            # Check start position
            dist = torch.linalg.norm(pos_tensor - m_pos_tensor)
            if dist < self.safety_margin:
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
            if dist_end < self.safety_margin:
                return False
            
            # Check along the patrol line (sample points between start and end)
            # Use 5 samples along the path
            for alpha in [0.25, 0.5, 0.75]:
                interpolated_pos = m_pos_tensor + alpha * (p_end - m_pos_tensor)
                dist_path = torch.linalg.norm(pos_tensor - interpolated_pos)
                if dist_path < self.safety_margin:
                    return False
                
        # Also check against the outer map boundary
        if abs(pos[0]) > self.world_x_bound or abs(pos[1]) > self.world_y_bound:
             return False
             
        return True


    def reset_world_at(self, env_index: int = None):
        # 1. Decide which Maps to use (Mixed Batch Logic)
        # If env_index is None when initializing at the start of training or evaluation(reset all), we pick randoms for everyone
        if env_index is None:
            # Resetting ALL envs.
            indices = torch.arange(self.world.batch_dim, device=self.world.device)
            # Reset time for all
            self.global_time.fill_(0.0)

            if self.training:
                # Randomly assign Map 0 or Map 1 to each env in the batch
                map_choices = torch.randint(0, 2, (self.world.batch_dim,), device=self.world.device)
            else:
                # Evaluation Mode: Force Map 2 (High Level)
                map_choices = torch.full((self.world.batch_dim,), 2, device=self.world.device)
        else:
            # Resetting SINGLE env
            indices = torch.tensor([env_index], device=self.world.device)
            # Reset time for this env
            self.global_time[env_index] = 0.0

            if self.training:
                map_choices = torch.randint(0, 2, (1,), device=self.world.device)
            else:
                map_choices = torch.tensor([2], device=self.world.device)


        # 2. Configure Maps (Iterate via CPU to handle mixed maps)
        # First, reset all entities to "Out of Bounds" for these indices
        # Fix: Use a position just outside the map bounds instead of [1000, 1000] to avoid camera zoom issues
        out_of_bounds = torch.tensor([self.world_x_bound + 1.0, self.world_y_bound + 1.0], device=self.world.device)
        for entity in self.obstacle_entities:
            entity.set_pos(out_of_bounds, batch_index=indices)
            # Also reset velocity and rotation for moving obstacles to prevent drift
            if hasattr(entity, 'state') and hasattr(entity.state, 'vel'):
                entity.state.vel[indices] = 0.0
            if hasattr(entity, 'state') and hasattr(entity.state, 'rot'):
                entity.state.rot[indices] = 0.0

        # Apply specific map configurations per environment
        cpu_map_choices = map_choices.tolist()
        cpu_indices = indices.tolist()
        grid_size = 0.2

        for i, map_id in zip(cpu_indices, cpu_map_choices):
            walls_data, moving_data = self.parsed_maps[map_id]
            
            # Set Visual Border (4 sides of the map)
            # Top border
            self.border_segments[0].set_pos(
                torch.tensor([0.0, self.world_y_bound], device=self.world.device), 
                batch_index=i
            )
            self.border_segments[0].set_rot(torch.tensor([0.0], device=self.world.device), batch_index=i)
            
            # Bottom border
            self.border_segments[1].set_pos(
                torch.tensor([0.0, -self.world_y_bound], device=self.world.device), 
                batch_index=i
            )
            self.border_segments[1].set_rot(torch.tensor([0.0], device=self.world.device), batch_index=i)
            
            # Left border
            self.border_segments[2].set_pos(
                torch.tensor([-self.world_x_bound, 0.0], device=self.world.device), 
                batch_index=i
            )
            self.border_segments[2].set_rot(torch.tensor([1.5708], device=self.world.device), batch_index=i)
            
            # Right border
            self.border_segments[3].set_pos(
                torch.tensor([self.world_x_bound, 0.0], device=self.world.device), 
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
                    # Fix: Access 'pos' from the dictionary
                    start_pos = torch.tensor(m_data['pos'], device=self.world.device)
                    obs.set_pos(start_pos, batch_index=i)
                    
                    # Randomly decide Patrol Type 
                    p_start = start_pos.clone()
                    p_end = start_pos.clone()
                    
                    otype = m_data['type']
                    if otype == 'U':
                        # Up/Down. Starts at map pos, goes down 4 cells.
                        p_end[1] += 5 * grid_size
                    elif otype == 'S':
                        # Sideways. Starts at map pos, goes right 4 cells.
                        p_end[0] += 5 * grid_size

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
                # 2. Sample distance (0.5m min to prevent trivial rewards, up to L)
                distance = np.random.uniform(0.5, self._max_dist)
                
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
                   final_dist < 0.2:
                    attempts += 1
                    continue

                # Placement successful
                self.agent.set_pos(torch.tensor(agent_pos_np, device=self.world.device), batch_index=i)
                self.goal.set_pos(torch.tensor(goal_pos_np, device=self.world.device), batch_index=i)
                placed = True

            if not placed:
                # Fallback in case of failure (should be rare with 1000 attempts)
                print(f"Warning: Failed to place agent/goal after 1000 attempts in environment {i}. Placing randomly.")
                # Fallback positions must also respect the 5.6m x 3.6m bounds
                self.agent.set_pos(torch.tensor([np.random.uniform(x_range[0], x_range[1]), np.random.uniform(y_range[0], y_range[1])], device=self.world.device).unsqueeze(0), batch_index=i)
                self.goal.set_pos(torch.tensor([np.random.uniform(x_range[0], x_range[1]), np.random.uniform(y_range[0], y_range[1])], device=self.world.device).unsqueeze(0), batch_index=i)



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
            
            if agent.action.u.shape[-1] > 1:
                action_index = torch.argmax(agent.action.u, dim=-1)
            else:
                action_index = agent.action.u.squeeze(-1).long()

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
            agent.state.rot[:] = rot
            agent.state.force[:] = 0.0

    def reward(self, agent: Agent):
        # Paper Reward:
        # +1 if reach goal
        # -1 if collision
        # -0.01 time penalty
        
        # Distance to goal
        dist = torch.linalg.norm(agent.state.pos - self.goal.state.pos, dim=-1)
        is_at_goal = dist < 0.2
        
        # Collision Check
        is_collision = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)
        for entity in self.world.agents + self.world.landmarks:
            if entity != agent and entity.collide:
                 is_collision = is_collision | self.world.is_overlapping(agent, entity)

        rews = torch.zeros_like(dist)
        rews[is_at_goal] += 1.0
        rews[is_collision] -= 1.0
        rews += -0.01 # Time penalty
        
        return rews

    def observation(self, agent: Agent):

        # Veuillez noter que vous devriez implémenter un mécanisme pour obtenir les angles pour être 100% conforme 
        # à l'étude (qui utilise Lidar signal et Local Target Position), mais la conversion ci-dessus est la 
        # manière la plus courante d'adapter un vecteur plat de 800 éléments à l'entrée CNN du rapport.

        # Lidar.measure() renvoie un vecteur de 360 distances (une par rayon)
        lidar_distances = agent.sensors[0].measure() # (B, 360)

        # 1. Calculer les coordonnées relatives au but (Delta X, Delta Y)
        # C'est la position locale du but vue par l'agent.
        delta_pos = self.goal.state.pos - agent.state.pos # (B, 2)
    
        # Séparer les angles et les distances Lidar (VMAS donne souvent les distances directement)
        # Pour respecter les 720 données (360 angles, 360 distances), nous devons créer l'information d'angle.
        # Dans VMAS, les 360 valeurs sont D, D, D... Nous allons devoir simplifier l'entrée à 360 distances pour ce test.
        # Pour respecter les 800 inputs de l'étude, nous allons concaténer 40 copies de (dX, dY).

        # 2. Créer l'entrée cible (40 copies de (dX, dY) = 80 données)
        local_target_input = delta_pos.repeat(1, 40) # (B, 80)
    
        # 3. Mettre l'observation Lidar au format (360 * 2 = 720) en dupliquant les distances
        # pour simuler l'entrée angle/distance si l'information d'angle n'est pas disponible.
        # Si lidar.measure() renvoie les distances pour les 360 rayons:
        lidar_input = torch.cat([lidar_distances, lidar_distances], dim=-1) # (B, 720)

        # 4. Concaténer pour obtenir 800 éléments
        raw_input_800 = torch.cat([lidar_input, local_target_input], dim=-1) # (B, 800)
    
        # 5. Redimensionner pour le CNN: (B, 800) -> (B, 2, 20, 20)
        # L'étude utilise (20x20x2) car (20*20*2 = 800)
        return raw_input_800.reshape(self.world.batch_dim, 2, 20, 20)

    def set_max_dist(self, max_dist: float):
        """Sets the maximum spawning distance for curriculum learning."""
        self._max_dist = max_dist
        if self.debug:
            print(f"[Scenario] Set spawn max_dist to: {self._max_dist}")

    @property
    def max_dist(self) -> float:
        """Gets the current maximum spawning distance."""
        return self._max_dist