import torch
import time
import numpy as np
from vmas import make_env
from vmas.simulator.core import Agent, Sphere
from vmas.simulator.utils import Color
from environment.path_planning import PathPlanningScenario
from agent.ddqn_agent import QNetwork
from agent.params import Params

# --- CONFIGURATION ---
MODEL_FOLDER = "results/251214-181350/"
MODEL_PATH = MODEL_FOLDER + "ddqn_q_network.pt"
HYPERPARAM_PATH = MODEL_FOLDER + "hyperparameters.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EVAL_EPISODES = 50
RENDER_DELAY = 0.05  # Delay in seconds between steps for visualization


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

    # 1. Create Environment using the CUSTOM VisualizerScenario
    # This ensures n_agents is 3 (Robot + 2 Anchors) from the start.
    env = make_env(
        scenario=VisualizerScenario(),
        num_envs=1,
        continuous_actions=False,
        max_steps=Params.MAX_STEPS_PER_EPISODE,
        device=DEVICE,
        # seed=0,
        dict_spaces=True,
        multidiscrete_actions=True,
    )

    # Set max difficulty to use full map
    env.scenario.set_max_dist(Params.L_MAX)

    # 2. Inspect Environment
    obs_dict = env.reset()
    state_tensor = obs_dict["robot"]

    if state_tensor.dim() == 4 and state_tensor.shape[0] == 1:
        state_tensor = state_tensor.squeeze(0)

    n_rays = state_tensor.shape[-1]
    print(f"Detected N_LIDAR_RAYS: {n_rays}")

    # 3. Load Network
    q_network = QNetwork(state_size=Params.CNN_INPUT_CHANNELS, action_size=Params.ACTION_SIZE).to(DEVICE)

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        q_network.load_state_dict(checkpoint)
        print(f"Successfully loaded model from: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"\nERROR: Could not find model file at: {MODEL_PATH}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    q_network.eval()

    # 4. Loop
    # for episode in range(NUM_EVAL_EPISODES):
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

        while not done:
            # Render with agent_index_focus=None.
            # VMAS will zoom out to fit the Robot AND the Anchors.
            env.render(
                mode="human",
                env_index=0,
                agent_index_focus=None
            )

            with torch.no_grad():
                q_values = q_network(state)
                action_idx = torch.argmax(q_values, dim=1)

            action_input = action_idx.unsqueeze(-1)

            # --- Dummy Actions for Anchors ---
            # Anchors expect (Batch, 2) continuous actions (forces). We send zeros.
            dummy_action = torch.zeros(1, 2, device=DEVICE)

            # Now we send 3 actions, matching env.n_agents
            next_obs, rewards, dones, info = env.step([action_input, dummy_action, dummy_action])

            next_state = next_obs["robot"]
            if next_state.dim() == 4 and next_state.shape[0] == 1:
                next_state = next_state.squeeze(0)

            reward = rewards["robot"]
            total_reward += reward.item()
            done = dones.any().item()

            state = next_state
            step_count += 1
            time.sleep(RENDER_DELAY)

        print(f"Episode {episode + 1} Finished. Total Reward: {total_reward:.2f}, Steps: {step_count}")


if __name__ == "__main__":
    load_hyperparameters(HYPERPARAM_PATH)
    evaluate_visual()
