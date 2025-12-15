import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from collections import deque
from .params import Params


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()

        # 1. Use the passed n_rays (from main_train.py), fallback to Params only if necessary
        self.n_rays = Params.N_LIDAR_RAYS

        # Standard 1D Convolutions for Lidar sequences
        self.conv1 = nn.Conv1d(in_channels=state_size, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)

        # Run a dummy pass to check output size.
        # This handles variable n_rays (0, 10, 360) without math errors.
        with torch.no_grad():
            dummy_input = torch.zeros(1, state_size, self.n_rays)

            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)

            self.flat_size = int(x.view(1, -1).size(1))

        self.fc1 = nn.Linear(self.flat_size, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, state):
        # state shape: (Batch, 4, N_RAYS)

        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = x.reshape(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    """
    Experience Replay Buffer.
    Stores: [s, a, r, s_next, done]
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        # IMPORTANT: Ensure inputs are CPU numpy arrays or python scalars to save VRAM
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def evaluate_agent(q_network: nn.Module, env, global_target_pos: np.ndarray, num_eval_episodes: int = 5):
    """
    Evaluates the agent using VMAS API.
    """
    q_network.eval()  # Set network to eval mode
    rewards_per_eval_episode = []

    # Target global tensor
    FINAL_GOAL_TENSOR = torch.tensor(global_target_pos, device=env.device, dtype=torch.float32)

    # We will use the first environment index (0) for evaluation
    eval_index = 0

    for episode in range(num_eval_episodes):
        # 1. Reset specific env index
        # VMAS reset_at returns nothing, it updates internal state. 
        # We must fetch obs separately or rely on reset_at signature if modified, 
        # but standard VMAS usage is reset -> get_obs.
        env.reset_at(eval_index)

        # Manually set goal to start position if needed, or rely on random spawn
        # Ideally, for consistent testing, you might want to force a specific spawn here.

        # Get initial observation
        obs_dict = env.get_from_scenario(get_observations=True, dict_agent_names=True)
        # Check if obs_dict is a list (some VMAS versions) or dict
        if isinstance(obs_dict, list):
            state_tensor = obs_dict[0]['robot'][eval_index].unsqueeze(0)  # (1, 2, 20, 20)
        else:
            state_tensor = obs_dict['robot'][eval_index].unsqueeze(0)

        total_reward = 0.0
        current_pos = env.scenario.agent.state.pos[eval_index]  # Direct access to pos

        # Reset local tracking
        prev_reward = -1.0
        done = False

        for step in range(Params.MAX_TEST_STEPS):
            # --- 2. High Level Planner Logic ---
            dist_to_final_goal = torch.linalg.norm(current_pos - FINAL_GOAL_TENSOR)

            if dist_to_final_goal < 0.2:
                total_reward += 10.0
                break

                # If we hit the local goal (reward ~ +10.0 in new scale, or +1.0 in old)
            # We check for > 0.5 as a generic "positive reward" threshold
            if step > 0 and prev_reward > 0.5:
                target_vec = FINAL_GOAL_TENSOR - current_pos
                target_norm = target_vec / (torch.linalg.norm(target_vec) + 1e-6)
                new_local_goal_pos = current_pos + target_norm * env.scenario.lidar_range

                # Update VMAS goal entity
                env.scenario.goal.set_pos(new_local_goal_pos.unsqueeze(0), batch_index=eval_index)

            # --- 3. Action Selection ---
            with torch.no_grad():
                q_values = q_network(state_tensor)
                action_idx = torch.argmax(q_values, dim=1)  # (1,)

            # --- 4. Step ---
            # VMAS expects a LIST of actions, one per agent.
            # We have 1 agent. The action must be shape (B, 1) or (B,) depending on scenario.
            # Our scenario expects (B, 1) for .u[:, 0]
            action_input = action_idx.unsqueeze(-1)  # (1, 1)

            # We step ALL environments, but we only care about index 0.
            # Ideally we'd have a separate eval env, but here we just step everything 
            # and ignore others (or pass dummies). 
            # For simplicity in this function, we assume env is just for eval or we tolerate stepping others.
            # To be safe, we construct a batch of actions.

            full_action = torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.long)
            full_action[eval_index] = action_input

            obs_dict, reward_dict, done_tensor, info_dict = env.step([full_action])

            # Extract data for our specific env index
            next_state_tensor = obs_dict['robot'][eval_index].unsqueeze(0)
            reward = reward_dict['robot'][eval_index].item()
            done = done_tensor[eval_index].item()

            total_reward += reward
            state_tensor = next_state_tensor
            current_pos = env.scenario.agent.state.pos[eval_index]
            prev_reward = reward

            if done:
                break

            # Collision termination check (if not handled by Done)
            if reward < -4.0:  # Assuming -5.0 is collision
                break

        rewards_per_eval_episode.append(total_reward)

    q_network.train()  # Reset to train mode
    average_reward = np.mean(rewards_per_eval_episode) if rewards_per_eval_episode else 0.0
    return rewards_per_eval_episode, average_reward


def update_L(n_steps):
    """
    Updates curriculum distance L.
    """
    if n_steps <= Params.N1_THRESHOLD:
        L = Params.L_MIN
    elif Params.N1_THRESHOLD < n_steps < Params.N2_THRESHOLD:
        L = Params.L_MIN + Params.M_SEARCH_SPEED * (n_steps - Params.N1_THRESHOLD)
    else:
        L = Params.L_MAX

    return L