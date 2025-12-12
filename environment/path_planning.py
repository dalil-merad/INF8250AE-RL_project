import torch
import numpy as np
from vmas.simulator.core import Agent, Landmark, World, Sphere, Box
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color
from vmas.simulator.dynamics.common import Dynamics

TIMESTEP_REWARD = -0.01
GOAL_REWARD = 10.0
COLLISION_REWARD = -10.0


class KinematicDynamics(Dynamics):
    @property
    def needed_action_size(self) -> int:
        return 1  # We expect 1D action input (action index)

    def process_action(self):
        pass

class PathPlanningScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.lidar_range = 2.0  # Lidar max range
        self.n_lidar_rays = 360  # Number of Lidar rays
        self.world_size = 2.0  # World is a square from -2 to 2
        self.agent_radius = 0.1  # Agent radius
        self.goal_radius = 0.1  # Goal radius

        self._max_dist = 4.0  # Default max distance

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
                entity_filter=lambda e: e.name.startswith("wall"),
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

        # ----- Create Borders (Walls) -----
        self.walls = []
        wall_configs = [
            (0, self.world_size, 2 * self.world_size, 0.1),   # Top
            (0, -self.world_size, 2 * self.world_size, 0.1),  # Bottom
            (-self.world_size, 0, 0.1, 2 * self.world_size),  # Left
            (self.world_size, 0, 0.1, 2 * self.world_size)    # Right
        ]

        for i, (x, y, l, w) in enumerate(wall_configs):
            wall = Landmark(
                name=f"wall_{i}",
                shape=Box(length=l, width=w),
                color=Color.RED,
                collide=True
            )
            # Set positions immediately
            world.add_landmark(wall)
            wall.set_pos(torch.tensor([x, y], device=device).unsqueeze(0).repeat(batch_dim, 1), batch_index=None)
            self.walls.append(wall)

        return world

    def reset_world_at(self, env_index: int = None):
        margin = 0.2
        limit = self.world_size - margin

        if env_index is None:
            batch_size = self.world.batch_dim
            # Randomize Goal
            random_goal_pos = (torch.rand((batch_size, 2), device=self.world.device) * 2 * limit) - limit
            self.goal.set_pos(random_goal_pos, batch_index=None)

            # Randomize Agent (Relative)
            theta = torch.rand((batch_size, 1), device=self.world.device) * 2 * np.pi
            dist = torch.rand((batch_size, 1), device=self.world.device) * self._max_dist

            rel_pos = torch.cat([torch.cos(theta), torch.sin(theta)], dim=1) * dist
            random_agent_pos = random_goal_pos + rel_pos
            random_agent_pos = torch.clamp(random_agent_pos, -limit, limit)

            self.agent.set_pos(random_agent_pos, batch_index=None)
            self.agent.set_vel(torch.zeros((batch_size, 2), device=self.world.device), batch_index=None)
        else:
            random_goal_pos = (torch.rand((1, 2), device=self.world.device) * 2 * limit) - limit
            self.goal.set_pos(random_goal_pos, batch_index=env_index)

            theta = torch.rand((1, 1), device=self.world.device) * 2 * np.pi
            dist = torch.rand((1, 1), device=self.world.device) * self._max_dist

            rel_pos = torch.cat([torch.cos(theta), torch.sin(theta)], dim=1) * dist
            random_agent_pos = random_goal_pos + rel_pos
            random_agent_pos = torch.clamp(random_agent_pos, -limit, limit)

            self.agent.set_pos(random_agent_pos, batch_index=env_index)
            self.agent.set_vel(torch.zeros((1, 2), device=self.world.device), batch_index=env_index)

    def reward(self, agent: Agent):
        step_reward = torch.full((self.world.batch_dim,), TIMESTEP_REWARD, device=self.world.device)

        dist_to_goal = torch.linalg.norm(agent.state.pos - self.goal.state.pos, dim=-1)
        at_goal = dist_to_goal < (self.agent_radius + self.goal_radius)
        step_reward[at_goal] += GOAL_REWARD

        is_collision = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)
        for wall in self.walls:
            is_collision |= self.world.is_overlapping(agent, wall)

        step_reward[is_collision] += COLLISION_REWARD
        return step_reward

    def done(self):
        dist_to_goal = torch.linalg.norm(self.agent.state.pos - self.goal.state.pos, dim=-1)
        at_goal = dist_to_goal < (self.agent_radius + self.goal_radius)

        is_collision = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)
        for wall in self.walls:
            is_collision |= self.world.is_overlapping(self.agent, wall)

        return at_goal | is_collision

    def observation(self, agent: Agent):
        # Shape: (B, 2, 20, 20)
        batch_dim = self.world.batch_dim
        device = self.world.device

        # 1. Lidar Data: Shape (B, 360)
        lidar_distances = agent.sensors[0].measure()

        # Create Angle channel: Shape (B, 360)
        lidar_angles = torch.linspace(0, 1, self.n_lidar_rays, device=device).unsqueeze(0).expand(batch_dim, -1)

        # Stack -> (B, 360, 2) -> Permute to (B, 2, 360)
        lidar_combined = torch.stack([lidar_distances, lidar_angles], dim=2).permute(0, 2, 1)

        # Reshape to (B, 2, 18, 20)
        lidar_reshaped = lidar_combined.reshape(batch_dim, 2, 18, 20)

        # 2. Relative Goal Position: Shape (B, 2)
        rel_pos = self.goal.state.pos - agent.state.pos

        # Expand to fill (B, 2, 2, 20)
        rel_pos_expanded = rel_pos.repeat(1, 40)  # (B, 80)
        target_reshaped = rel_pos_expanded.reshape(batch_dim, 2, 2, 20)

        # 3. Concatenate along height (dim 2)
        obs = torch.cat([lidar_reshaped, target_reshaped], dim=2)

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
