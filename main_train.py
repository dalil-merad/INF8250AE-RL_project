from environment.path_planning import PathPlanningScenario # Assurez-vous d'importer la bonne classe
from agent.ddqn_agent import QNetwork, ReplayBuffer, update_L, evaluate_agent
from agent.params import * # Importer tous les hyperparamètres
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from vmas import make_env

CNN_INPUT_CHANNELS = 2
ACTION_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_step = 0
epsilon = EPSILON_START

# Initialisation de l'environnement VMAS (PathPlanningScenario)
env = PathPlanningScenario()
env.make_world(
    batch_dim=1, 
    device=DEVICE, 
    training=True,      # Mode entraînement (Map 0/1)
    max_dist=L_MIN      # Initialisation de la distance L (Curriculum)
)
# Initialisation des réseaux (CNN)
q_network = QNetwork(state_size=CNN_INPUT_CHANNELS, action_size=ACTION_SIZE).to(DEVICE)
target_q_network = QNetwork(state_size=CNN_INPUT_CHANNELS, action_size=ACTION_SIZE).to(DEVICE)
target_q_network.load_state_dict(q_network.state_dict()) # Initialisation du réseau cible
target_q_network.eval() # Le réseau cible ne doit pas être en mode entraînement

# Autres initialisations
optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()
replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
rewards_per_episode = []


# --- 3. Boucle d'Entraînement Complète (Logique DDQN) ---
for episode in range(NUM_EPISODES):
    # A. Curriculum Learning: Mettre à jour L et l'appliquer à l'environnement
    max_dist_L = update_L(current_step)
    env.set_max_dist(max_dist_L)

    # B. Réinitialisation de l'environnement (Map 0 ou 1)
    state_tensor_800, info = env.reset()
    state_np = state_tensor_800.cpu().numpy().squeeze(0) # Forme: (2, 20, 20)
    
    total_reward = 0
    
    for step in range(MAX_STEPS_PER_EPISODE):
        current_step += 1
        
        # C. Sélection de l'action (Epsilon-greedy)
        if random.random() < epsilon:
            action = env.action_space.sample()  
        else:
            with torch.no_grad():
                # state_np est déjà (2, 20, 20), on ajoute la dimension de batch (1)
                state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                q_values = q_network(state_tensor)
                action = torch.argmax(q_values).item()
        
        # D. Exécuter l'action et stocker
        next_state_tensor_800, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        next_state_np = next_state_tensor_800.cpu().numpy().squeeze(0)
        
        replay_buffer.add(state_np, action, reward, next_state_np, done)
        
        # E. Entraînement par lot (DDQN)
        if len(replay_buffer) > BATCH_SIZE and current_step % TRAINING_FREQUENCY == 0:
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

            # Conversion du lot en Tenseurs (Shape: BATCH_SIZE, 2, 20, 20)
            states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE)
            actions_tensor = torch.tensor(actions, dtype=torch.long).to(DEVICE)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
            next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE)
            dones_tensor = torch.tensor(dones, dtype=torch.float32).to(DEVICE)

            # --- CALCUL DDQN ---
            with torch.no_grad():
                actions_from_q_network = torch.argmax(q_network(next_states_tensor), dim=1).unsqueeze(1)
                next_q_values_target = target_q_network(next_states_tensor).gather(1, actions_from_q_network).squeeze(1)
                target_q_values = rewards_tensor + GAMMA * next_q_values_target * (1 - dones_tensor)

            predicted_q_values = q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

            loss = criterion(predicted_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Mise à jour Epsilon
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        state_np = next_state_np # Passer à l'état suivant
        
        # F. Conditions d'arrêt
        if done or reward < -0.5: 
            break
            
    # G. Mise à jour du réseau cible
    if (episode + 1) % TARGET_UPDATE_FREQUENCY == 0:
        target_q_network.load_state_dict(q_network.state_dict())

    rewards_per_episode.append(total_reward)
    
    # [Ajouter ici la logique d'affichage des récompenses]
    
# --- 4. Évaluation Finale ---
# Global Target Pos est la destination finale du parcours test (e.g., [4.0, 1.0])
# GLOBAL_TARGET_POS = np.array([4.0, 1.0]) 
# eval_rewards, avg_reward = evaluate_agent(q_network, env, GLOBAL_TARGET_POS, num_eval_episodes=5)
# print(f"Average reward during evaluation: {avg_reward:.2f}")