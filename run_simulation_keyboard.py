import torch
from vmas import render_interactively
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

    # 2. Create the World
    world = scenario.make_world(
        batch_dim=BATCH_DIM, 
        device=DEVICE,
        dt=DT
    )

    # 3. Call the interactive renderer
    print(f"Starting V-MAS visualization for {BATCH_DIM} environment(s) on device: {DEVICE}...")
    print("Controls:")
    print("  Arrow Right : +X velocity")
    print("  Arrow Left  : -X velocity")
    print("  Arrow Up    : +Y velocity")
    print("  Arrow Down  : -Y velocity")
    print("  M / N       : Rotation (if enabled)")
    
    # render_interactively runs the simulation loop and displays the world.
    # We pass the scenario and world objects.
    render_interactively(
        scenario=scenario,
        world=world,
        # The agent's policy is set to 'user' which allows you to manually control 
        # the robot using the arrow keys (or WASD) during the visualization.
        agent_policies='user' 
    )

if __name__ == '__main__':

    run_visualization()