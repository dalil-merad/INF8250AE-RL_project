import torch
import time
import numpy as np
from vmas import make_env
from vmas.simulator.core import Agent, Sphere
from vmas.simulator.utils import Color
from environment.path_planning import PathPlanningScenario
from agent.ddqn_agent import QNetwork
from agent.params import Params
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import os

# --- CONFIGURATION ---
MODEL_FOLDER = "results/251214-181350/"
MODEL_PATH = MODEL_FOLDER + "ddqn_q_network.pt"
HYPERPARAM_PATH = MODEL_FOLDER + "hyperparameters.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EVAL_EPISODES = 50
RENDER_DELAY = 0.05  # Delay in seconds between steps for visualization
EXPORT_TRAJECTORY = True  # Whether to export trajectory plots
LIVE_DISPLAY = True  # Whether to show live rendering


def load_hyperparameters(file_path):
    """
    Charge les hyperparamètres depuis un fichier texte et les applique à la classe Params.

    Args:
        file_path (str): Le chemin du fichier contenant les hyperparamètres.
    """
    try:
        with open(file_path, "r") as f:
            for line in f:
                key, value = line.strip().split(": ")
                # Convertir la valeur en float ou int si possible
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Garder comme string si la conversion échoue
                setattr(Params, key, value)
        print(f"\nHyperparamètres chargés avec succès depuis: {file_path}")
    except Exception as e:
        print(f"\nErreur lors du chargement des hyperparamètres: {e}")


def plot_trajectory(trajectory, env, episode_idx, output_folder):
    """
    Plots the agent's trajectory, goal, and obstacles (walls/borders) and saves to file.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. Setup Grid and Limits
    world_size = Params.WORLD_SIZE
    ax.set_xlim(-world_size - 0.5, world_size + 0.5)
    ax.set_ylim(-world_size - 0.5, world_size + 0.5)
    ax.set_aspect('equal')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.set_title(f"Episode {episode_idx} Trajectory")

    # 2. Draw Walls and Borders
    # Combine lists to iterate easily
    obstacles = env.scenario.walls + env.scenario.border_segments

    for obs in obstacles:
        # Get data for env_index 0
        pos = obs.state.pos[0].cpu().numpy()
        rot = obs.state.rot[0].item()  # Radians
        length = obs.shape.length
        width = obs.shape.width

        # Create Rectangle centered at (0,0) then transform
        # Matplotlib Rectangle defined by bottom-left corner
        rect = patches.Rectangle(
            (-length / 2, -width / 2),
            length, width,
            color='black' if 'border' in obs.name else 'red',
            alpha=0.6
        )

        # Apply rotation and translation
        t = Affine2D().rotate(rot).translate(pos[0], pos[1]) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

    # 3. Draw Goal
    goal_pos = env.scenario.goal.state.pos[0].cpu().numpy()
    goal_circle = patches.Circle(goal_pos, radius=Params.GOAL_RADIUS, color='green', alpha=0.7, label='Goal')
    ax.add_patch(goal_circle)

    # 4. Draw Trajectory
    traj = np.array(trajectory)
    if len(traj) > 0:
        # Plot path line
        ax.plot(traj[:, 0], traj[:, 1], color='blue', linewidth=2, label='Path', alpha=0.8)
        # Plot start point
        ax.scatter(traj[0, 0], traj[0, 1], color='blue', marker='o', s=50, label='Start')
        # Plot end point
        ax.scatter(traj[-1, 0], traj[-1, 1], color='purple', marker='x', s=50, label='End')

    ax.legend(loc='upper right')

    # 5. Save
    save_path = os.path.join(output_folder, f"trajectory_ep_{episode_idx}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Trajectory plot saved to: {save_path}")


# --- CUSTOM SCENARIO FOR VISUALIZATION ---
class VisualizerScenario(PathPlanningScenario):
    """
    Extends the base scenario to add invisible Camera Anchors.
    This forces VMAS to keep the camera zoomed out to fit the whole map.
    """

    def make_world(self, batch_dim, device, **kwargs):
        # 1. Create the standard world (Robot + Map)
        world = super().make_world(batch_dim, device, **kwargs)

        # 2. Add Anchor Agents
        # We add them here so VMAS counts them in n_agents (Total = 3)
        self.anchors = []
        for name in ["anchor_bl", "anchor_tr"]:
            anchor = Agent(
                name=name,
                shape=Sphere(0.05),
                collide=False,
                color=Color.BLACK,
                sensors=[],  # No sensors -> No Lidar crash
                u_range=0.0  # No movement capability
            )
            world.add_agent(anchor)
            self.anchors.append(anchor)

        return world

    def reset_world_at(self, env_index=None):
        # 1. Reset Robot/Map using parent logic
        super().reset_world_at(env_index)

        # 2. Force Anchors to corners
        bx = self.world_size
        by = self.world_size
        margin = 0.5

        # Bottom-Left Anchor
        self.anchors[0].set_pos(
            torch.tensor([-bx - margin, -by - margin], device=self.world.device).unsqueeze(0).repeat(
                self.world.batch_dim, 1),
            batch_index=env_index
        )
        self.anchors[0].set_vel(torch.zeros(2, device=self.world.device), batch_index=env_index)

        # Top-Right Anchor
        self.anchors[1].set_pos(
            torch.tensor([bx + margin, by + margin], device=self.world.device).unsqueeze(0).repeat(self.world.batch_dim,
                                                                                                   1),
            batch_index=env_index
        )
        self.anchors[1].set_vel(torch.zeros(2, device=self.world.device), batch_index=env_index)

    def observation(self, agent: Agent):
        # Override to handle Anchors (which have no sensors)
        if agent.name == "robot":
            return super().observation(agent)
        else:
            # Return dummy observation for anchors
            return torch.zeros(1, device=self.world.device)

    def reward(self, agent: Agent):
        # Override to prevent errors if anchors are queried for reward
        if agent.name == "robot":
            return super().reward(agent)
        return torch.zeros(self.world.batch_dim, device=self.world.device)


def evaluate_visual():
    print(f"--- Initializing Environment for Visualization ---")

    # Ensure output folder exists for images
    img_folder = os.path.join(MODEL_FOLDER, "trajectory_plots")
    os.makedirs(img_folder, exist_ok=True)

    # [Setup code remains the same...]
    env = make_env(
        scenario=VisualizerScenario(),
        num_envs=1,
        continuous_actions=False,
        max_steps=Params.MAX_STEPS_PER_EPISODE,
        device=DEVICE,
        dict_spaces=True,
        multidiscrete_actions=True,
    )
    env.scenario.set_max_dist(Params.L_MAX)

    # [Network loading code remains the same...]
    q_network = QNetwork(state_size=Params.CNN_INPUT_CHANNELS, action_size=Params.ACTION_SIZE).to(DEVICE)
    # ... load weights ...
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    q_network.load_state_dict(checkpoint)
    q_network.eval()

    episode = 0
    while True:
        episode += 1
        print(f"\nStarting Episode {episode}...")

        obs_dict = env.reset()
        state = obs_dict["robot"]
        if state.dim() == 4 and state.shape[0] == 1:
            state = state.squeeze(0)

        done = False
        total_reward = 0
        step_count = 0

        trajectory = []

        while not done:
            # 1. Record current position BEFORE stepping
            current_pos = env.scenario.agent.state.pos[0].cpu().numpy().copy()
            trajectory.append(current_pos)

            # Optional: Render live window
            if LIVE_DISPLAY:
                env.render(mode="human", env_index=0, agent_index_focus=None)

            with torch.no_grad():
                q_values = q_network(state)
                action_idx = torch.argmax(q_values, dim=1)

            action_input = action_idx.unsqueeze(-1)
            dummy_action = torch.zeros(1, 2, device=DEVICE)

            # Step
            next_obs, rewards, dones, info = env.step([action_input, dummy_action, dummy_action])

            # ... update logic ...
            next_state = next_obs["robot"]
            if next_state.dim() == 4 and next_state.shape[0] == 1:
                next_state = next_state.squeeze(0)

            state = next_state
            total_reward += rewards["robot"].item()
            done = dones.any().item()
            step_count += 1

            if LIVE_DISPLAY:
                time.sleep(RENDER_DELAY)

        if EXPORT_TRAJECTORY:
            # We append the final position to complete the line
            final_pos = env.scenario.agent.state.pos[0].cpu().numpy().copy()
            trajectory.append(final_pos)

            plot_trajectory(trajectory, env, episode, img_folder)

        print(f"Episode {episode} Finished. Total Reward: {total_reward:.2f}")

        # Break after N episodes if desired
        if episode >= NUM_EVAL_EPISODES:
            break


if __name__ == "__main__":
    load_hyperparameters(HYPERPARAM_PATH)
    evaluate_visual()
