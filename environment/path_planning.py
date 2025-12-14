import torch
import numpy as np
from vmas.simulator.core import Agent, Landmark, World, Sphere, Box
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color
from vmas.simulator.dynamics.common import Dynamics
from agent.params import Params

MAX_STATIC_WALLS = 4  # Maximum number of static walls in the environment
SIZE_OF_WALLS = 6 / 5  # (* world_size) length of walls
WALL_COMPLEMENT = (Params.WORLD_SIZE - (SIZE_OF_WALLS * Params.WORLD_SIZE) / 2) / Params.WORLD_SIZE


class KinematicDynamics(Dynamics):
    @property
    def needed_action_size(self) -> int:
        return 1  # We expect 1D action input (action index)

    def process_action(self):
        pass


class WallConfigs:
    # Format: list of tuples (x, y, length, width)
    # Map is approx 8x8 meters (-4 to 4)

    # 1. Empty (No interior walls)
    empty = []

    # 2. Horizontal (Two parallel horizontal walls)
    h = [
        (-WALL_COMPLEMENT * Params.WORLD_SIZE, -.5 * Params.WORLD_SIZE, 6.0, 0.3),  # Bottom left
        (-WALL_COMPLEMENT * Params.WORLD_SIZE, .5 * Params.WORLD_SIZE, 6.0, 0.3),    # Top left
        (WALL_COMPLEMENT * Params.WORLD_SIZE, 0 * Params.WORLD_SIZE, 6.0, 0.3),    # Middle right
    ]

    # 3. Vertical (Two parallel vertical walls)
    v = [
        (-.5 * Params.WORLD_SIZE, -WALL_COMPLEMENT * Params.WORLD_SIZE, .3, 6.),  # Bottom left
        (0 * Params.WORLD_SIZE, WALL_COMPLEMENT * Params.WORLD_SIZE, .3, 6.),  # Top left
        (.5 * Params.WORLD_SIZE, -WALL_COMPLEMENT * Params.WORLD_SIZE, .3, 6.),  # Middle right
    ]

    # 4. Cross (Central intersection)
    c = [
        (0.0, 0.0, 7.0, 0.3),  # Horizontal long bar
        (0.0, 0.0, 0.3, 7.0)  # Vertical long bar
    ]

    # 5. Anti-cross
    ac = [
        (0.0, -Params.WORLD_SIZE, 0.3, 4.0),
        (0.0, Params.WORLD_SIZE, 0.3, 4.0),
        (-Params.WORLD_SIZE, 0.0, 4.0, 0.3),
        (Params.WORLD_SIZE, 0.0, 4.0, 0.3),
    ]

    # Helper to get all layouts
    @staticmethod
    def get_all():
        return [WallConfigs.empty, WallConfigs.h, WallConfigs.v, WallConfigs.c, WallConfigs.ac]

class PathPlanningScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.lidar_range = Params.LIDAR_RANGE  # Lidar max range
        self.n_lidar_rays = Params.N_LIDAR_RAYS  # Number of Lidar rays
        self.world_size = Params.WORLD_SIZE  # World is a square from -2 to 2
        self.agent_radius = Params.AGENT_RADIUS  # Agent radius
        self.goal_radius = Params.GOAL_RADIUS  # Goal radius

        self._max_dist = 4.0  # Default max distance
        self.max_static_walls = MAX_STATIC_WALLS  # Max number of static walls
        self.global_time = torch.empty(1)

    def set_max_dist(self, max_dist: float):
        self._max_dist = max_dist

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        world = World(batch_dim, device, dt=0.1)

        # ----- Create Agent -----
        self.agent = Agent(
            name="robot",
            shape=Sphere(radius=self.agent_radius),
            color=Color.BLUE,
            u_range=1.0,
            action_size=1,             # send index
            discrete_action_nvec=[8],  # Allow indices 0-7
            dynamics=KinematicDynamics(),  # custom dynamics for size 1 action (the actio index)
            sensors=[Lidar(
                world,
                n_rays=self.n_lidar_rays,
                max_range=self.lidar_range,
                entity_filter=lambda e: e.collide,
                render=True,
            )],
        )
        world.add_agent(self.agent)

        # ----- Create Goal -----
        self.goal = Landmark(
            name="goal",
            shape=Sphere(radius=self.goal_radius),
            color=Color.GREEN,
            collide=False
        )
        world.add_landmark(self.goal)

        # obstacle walls
        self.walls = []
        for i in range(self.max_static_walls):
            wall = Landmark(
                name=f"wall_{i}",
                shape=Box(length=(SIZE_OF_WALLS * self.world_size), width=0.2),  # Default size, changed in reset
                color=Color.RED,
                collide=True
            )
            world.add_landmark(wall)
            self.walls.append(wall)

        self.obstacle_entities = self.walls  # + self.moving_obstacles  # (if any)

        # World borders (collidable)
        self.border_segments = []
        # We'll create 4 thin rectangles for the border with fixed dimensions
        border_thickness = 0.02

        # Horizontal borders (top/bottom) - length matches map width
        for i in range(2):
            border = Landmark(
                name=f"border_h_{i}",
                shape=Box(length=(2 * self.world_size), width=border_thickness),
                color=Color.BLACK,
                collide=True,
            )
            world.add_landmark(border)
            self.border_segments.append(border)

        for i in range(2):
            border = Landmark(
                name=f"border_v_{i}",
                shape=Box(length=(2 * self.world_size), width=border_thickness),
                color=Color.BLACK,
                collide=True,
            )
            world.add_landmark(border)
            self.border_segments.append(border)

        # # ----- Create Borders (Walls) -----
        # self.walls = []
        # # (x, y, length, width)
        # wall_configs = [
        #     (-self.world_size, -self.world_size, 2 * self.world_size, 0.1),   # Top
        #     (-self.world_size, self.world_size, 2 * self.world_size, 0.1),  # Bottom
        #     (-self.world_size, -self.world_size, 0.1, 2 * self.world_size),  # Left
        #     (self.world_size, -self.world_size, 0.1, 2 * self.world_size)    # Right
        # ]
        #
        # for i, (x, y, l, w) in enumerate(wall_configs):
        #     wall = Landmark(
        #         name=f"wall_{i}",
        #         shape=Box(length=l, width=w),
        #         color=Color.RED,
        #         collide=True
        #     )
        #     # Set positions immediately
        #     world.add_landmark(wall)
        #     wall.set_pos(torch.tensor([x, y], device=device).unsqueeze(0).repeat(batch_dim, 1), batch_index=None)
        #     self.walls.append(wall)
        self.global_time = torch.zeros(batch_dim, device=device)

        return world

    def reset_world_at(self, env_index: int = None):
        # Determine indices to reset
        if env_index is None:
            indices = range(self.world.batch_dim)
        else:
            indices = [env_index]

        # Reset Time
        for i in indices:
            self.global_time[i] = 0.0

        # --- 1. CHOOSE AND PLACE WALLS ---
        layouts = WallConfigs.get_all()

        # Iterate over each environment to reset
        for i in indices:
            # Pick a random layout index (0=Empty, 1=H, 2=V, 3=Cross, 4=Anti-cross)
            layout_idx = np.random.randint(len(layouts))
            walls_data = layouts[layout_idx]

            # First, hide ALL interior walls by moving them far away
            # This handles "Empty" maps or maps with fewer walls than self.max_walls
            out_of_bounds = torch.tensor([100.0, 100.0], device=self.world.device)
            for wall in self.walls:
                wall.set_pos(out_of_bounds.unsqueeze(0), batch_index=i)

            # Place Active Walls from the chosen Layout
            for w_idx, (x, y, l, w) in enumerate(walls_data):
                if w_idx < len(self.walls):
                    wall = self.walls[w_idx]

                    # NOTE: In VMAS, we cannot change wall.shape.length per-environment
                    # because the shape object is shared across the batch.
                    # Instead, we interpret the config to set Rotation.

                    pos = torch.tensor([x, y], device=self.world.device).unsqueeze(0)
                    wall.set_pos(pos, batch_index=i)

                    # Logic: If config says Width > Length, it's a Vertical wall -> Rotate 90 deg
                    if w > l:
                        rot = torch.tensor([1.5708], device=self.world.device).unsqueeze(0)  # 90 deg
                        wall.set_rot(rot, batch_index=i)
                    else:
                        rot = torch.tensor([0.0], device=self.world.device).unsqueeze(0)  # 0 deg
                        wall.set_rot(rot, batch_index=i)

        # --- 2. PLACE AGENT AND GOAL ---
            self._place_agent_and_goal(i)

            self.border_segments[0].set_pos(
                torch.tensor([0.0, self.world_size], device=self.world.device),
                batch_index=i
            )
            self.border_segments[0].set_rot(torch.tensor([0.0], device=self.world.device), batch_index=i)

            # Bottom border
            self.border_segments[1].set_pos(
                torch.tensor([0.0, -self.world_size], device=self.world.device),
                batch_index=i
            )
            self.border_segments[1].set_rot(torch.tensor([0.0], device=self.world.device), batch_index=i)

            # Left border
            self.border_segments[2].set_pos(
                torch.tensor([-self.world_size, 0.0], device=self.world.device),
                batch_index=i
            )
            self.border_segments[2].set_rot(torch.tensor([1.5708], device=self.world.device), batch_index=i)

            # Right border
            self.border_segments[3].set_pos(
                torch.tensor([self.world_size, 0.0], device=self.world.device),
                batch_index=i
            )
            self.border_segments[3].set_rot(torch.tensor([1.5708], device=self.world.device), batch_index=i)

    def _place_agent_and_goal(self, env_index):
        # Simple rejection sampling to place agent/goal in free space
        placed = False
        attempts = 0
        limit = self.world_size - 0.5  # Keep away from borders

        while not placed and attempts < 100:
            # 1. Generate Random Goal
            goal_pos = (torch.rand(2, device=self.world.device) * 2 * limit) - limit

            # 2. Generate Random Agent near Goal
            angle = torch.rand(1, device=self.world.device) * 2 * np.pi
            dist = torch.rand(1, device=self.world.device) * self._max_dist
            dist = torch.clamp(dist, 0.2, self._max_dist)  # Min dist 0.2

            direction = torch.tensor([torch.cos(angle), torch.sin(angle)], device=self.world.device)
            # Ensure dimensions match for addition: goal_pos is [2], direction*dist is [2,1] usually
            agent_pos = goal_pos + (direction * dist).squeeze()

            # 3. Check Bounds (World limits)
            if (torch.abs(agent_pos) > limit).any() or (torch.abs(goal_pos) > limit).any():
                attempts += 1
                continue

            # 4. Set Positions Tentatively
            # We must set them now so the physics engine can check for overlaps below
            self.agent.set_pos(agent_pos.unsqueeze(0), batch_index=env_index)
            self.agent.set_vel(torch.zeros(2, device=self.world.device).unsqueeze(0), batch_index=env_index)
            self.goal.set_pos(goal_pos.unsqueeze(0), batch_index=env_index)

            # 5. Check Collision with Walls and Borders
            has_collision = False
            # Check against all walls (inactive ones are at 100,100 so they won't trigger)
            # Check against borders as well
            all_obstacles = self.walls + self.border_segments

            for obs in all_obstacles:
                # Check Agent
                if self.world.is_overlapping(self.agent, obs)[env_index]:
                    has_collision = True
                    break
                # Check Goal (optional, but good practice to ensure goal isn't inside a wall)
                if self.world.is_overlapping(self.goal, obs)[env_index]:
                    has_collision = True
                    break

            if has_collision:
                attempts += 1
                continue

            # If we reached here, positions are valid
            placed = True

        if not placed:
            print(f"Warning: Could not place agent safely in env {env_index} after {attempts} attempts.")

    def reward(self, agent: Agent):
        step_reward = torch.full((self.world.batch_dim,), Params.TIMESTEP_REWARD, device=self.world.device)

        dist_to_goal = torch.linalg.norm(agent.state.pos - self.goal.state.pos, dim=-1)
        at_goal = dist_to_goal < (self.agent_radius + self.goal_radius)
        step_reward[at_goal] += Params.GOAL_REWARD

        is_collision = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)
        for wall in self.walls:
            is_collision |= self.world.is_overlapping(agent, wall)

        step_reward[is_collision] += Params.COLLISION_REWARD
        return step_reward

    def done(self):
        dist_to_goal = torch.linalg.norm(self.agent.state.pos - self.goal.state.pos, dim=-1)
        at_goal = dist_to_goal < (self.agent_radius + self.goal_radius)

        is_collision = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)
        for wall in self.walls:
            is_collision |= self.world.is_overlapping(self.agent, wall)

        return at_goal | is_collision

    def observation(self, agent: Agent):
        batch_dim = self.world.batch_dim
        device = self.world.device
        n_rays = Params.N_LIDAR_RAYS

        # 1. Lidar Data: (B, 2, N_RAYS)
        # Channel 0: Distances, Channel 1: Angles
        lidar_distances = agent.sensors[0].measure()
        lidar_angles = torch.linspace(0, 1, n_rays, device=device).unsqueeze(0).expand(batch_dim, -1)

        # Shape: (B, 2, N_RAYS)
        lidar_data = torch.stack([lidar_distances, lidar_angles], dim=1)

        # 2. Goal Data: (B, 2)
        # We repeat the goal across all rays so the network can "compare" every ray to the goal.
        rel_pos = self.goal.state.pos - agent.state.pos
        # Shape: (B, 2, N_RAYS)
        goal_data = rel_pos.unsqueeze(2).expand(-1, -1, n_rays)

        # 3. Combine: (B, 4, N_RAYS)
        # Now you have 4 channels: [Distance, Angle, Goal_X, Goal_Y]
        obs = torch.cat([lidar_data, goal_data], dim=1)

        return obs

    def process_action(self, agent: Agent):
        if agent.name == "robot":
            # Robot speed is calculated to ensure a fixed 0.1m displacement per step.
            # Speed = (0.1m) / dt. Current dt=0.1, so speed=1.0.
            speed = 0.1 / self.world.dt
            diag_speed = speed * 0.707

            u = agent.action.u[:, 0]  # (B,), values in [-1, 1]
            n = 8  # Number of discrete actions

            # Continuous [-1, 1] -> Discrete [0, 7]
            action_index = torch.round((u + 1) / 2 * (n - 1)).long().clamp(0, n - 1)

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
            mask_4 = (action_index == 4)  # FL
            vx[mask_4] = -diag_speed
            vy[mask_4] = diag_speed
            rot[mask_4] = 2.35

            mask_5 = (action_index == 5)  # FR
            vx[mask_5] = diag_speed
            vy[mask_5] = diag_speed
            rot[mask_5] = 0.78

            mask_6 = (action_index == 6)  # BL
            vx[mask_6] = -diag_speed
            vy[mask_6] = -diag_speed
            rot[mask_6] = -2.35

            mask_7 = (action_index == 7)  # BR
            vx[mask_7] = diag_speed
            vy[mask_7] = -diag_speed
            rot[mask_7] = -0.78

            # Revert to Direct Velocity Control (Kinematic)
            agent.state.vel[:, 0] = vx
            agent.state.vel[:, 1] = vy
            agent.state.rot[:, 0] = rot
            # Zero out forces so physics doesn't add extra acceleration
            agent.state.force[:] = 0.0
