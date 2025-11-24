import torch
import numpy as np
from vmas import make_env
from vmas.interactive_rendering import InteractiveEnv
from scenarios.path_planning import PathPlanningScenario # Import your scenario class

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_DIM = 1
DT = 0.1 

class InteractivePathPlanningScenario(PathPlanningScenario):
    def process_action(self, agent):
        # Override robot control to handle continuous keyboard input
        if agent.name == "robot":
            # 1. Increment time (Critical for obstacles)
            self.global_time += self.world.dt
            
            # 2. Map continuous keyboard input to velocity
            # agent.action.u is [x, y] with range [-1, 1] from arrow keys
            speed = 0.1 / self.world.dt
            
            # Robustly handle action dimensions (VMAS maps Arrows->0,1 and M/N->2)
            u = agent.action.u
            
            # Fix: Clamp input to [-1, 1] to prevent explosion if u_range is set high
            u = torch.clamp(u, -1.0, 1.0)
            
            if u.shape[-1] >= 2:
                # Revert to Direct Velocity Control
                agent.state.vel[:, 0] = u[:, 0] * speed
                agent.state.vel[:, 1] = u[:, 1] * speed
                agent.state.force[:] = 0.0
            
            # Optional: Rotate to face movement direction
            if torch.linalg.norm(u[:, :2]) > 1e-3:
                agent.state.rot[:] = torch.atan2(u[:, 1], u[:, 0]).unsqueeze(-1)
            
            return

        # Delegate to original logic for obstacles
        super().process_action(agent)

def run_visualization():
    # 1. Instantiate the Scenario
    scenario = InteractivePathPlanningScenario()

    # 2. Setup Environment manually to control the Seed
    print(f"Starting V-MAS visualization for {BATCH_DIM} environment(s) on device: {DEVICE}...")
    print("Controls:")
    print("  Arrow Right : +X velocity")
    print("  Arrow Left  : -X velocity")
    print("  Arrow Up    : +Y velocity")
    print("  Arrow Down  : -Y velocity")
    print("  M / N       : Rotation (if enabled)")
    
    # Fix: render_interactively hardcodes seed=0. We use make_env + InteractiveEnv manually to randomize.
    env = make_env(
        scenario=scenario,
        num_envs=BATCH_DIM,
        device=DEVICE,
        continuous_actions=True,
        wrapper="gym",
        seed=np.random.randint(0, 100000), # Random seed for random positions
        wrapper_kwargs={"return_numpy": False},
        # Environment specific variables
        dt=DT,
        agent_policies='user',
        training=False,
        debug_prints=True 
    )

    # 3. Launch interactive viewer
    InteractiveEnv(
        env=env,
        control_two_agents=False,
        display_info=True,
        save_render=False,
        render_name="interactive"
    )

if __name__ == '__main__':

    run_visualization()