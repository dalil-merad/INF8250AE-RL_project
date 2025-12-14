from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from .params import Params


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(state_size, 16, kernel_size=2, stride=2)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 128, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, state):
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    """
    Mémoire d'expérience pour stocker les transitions (état, action, récompense, état_suivant, terminé).
    Utilise deque pour maintenir une taille maximale et gérer la politique FIFO (First-In First-Out).
    """
    def __init__(self, capacity):
        # La taille du buffer est définie par le rapport à 40 000 échantillons 
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Ajoute une transition au buffer.
        L'expérience se compose de [s, a, r, s₁, d][cite: 222].
        """
        # Note: 'state' et 'next_state' doivent être des numpy arrays ou des listes
        # pour éviter de stocker des tenseurs PyTorch lourds dans la mémoire.
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Échantillonne aléatoirement un lot (mini-batch) de transitions pour l'entraînement.
        """
        if len(self.buffer) < batch_size:
            return None 

        batch = random.sample(self.buffer, batch_size)
        
        # Déballer le lot en tuples
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Le reste de la conversion en tenseurs PyTorch se fera dans la boucle d'entraînement
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Retourne la taille actuelle de la mémoire.
        """
        return len(self.buffer)
    
def evaluate_agent(q_network: nn.Module, env, global_target_pos: np.ndarray, num_eval_episodes: int = 5):
    """
    Évalue la performance de l'agent DDQN sur la carte de test (Map 2) en naviguant vers une cible globale.

    :param q_network: Le réseau Q entraîné.
    :param env: L'instance PathPlanningScenario.
    :param global_target_pos: La position finale (x, y) de la navigation globale.
    :param num_eval_episodes: Nombre d'épisodes de test à exécuter.
    """
    rewards_per_eval_episode = []
    
    # Convertir la cible globale finale en tenseur pour la comparaison dans la boucle
    FINAL_GOAL_TENSOR = torch.tensor(global_target_pos, device=env.world.device, dtype=torch.float32)

    for episode in range(num_eval_episodes):
        # 1. Configuration de l'environnement de test
        # Assurez-vous que l'initialisation bascule en mode évaluation (Map 2)
        env.training = False 
        
        # Le reset place l'agent et la CIBLE LOCALE INITIALE (self.goal) dans la zone de test (Map 2)
        state_tensor_800, info = env.reset()  # state_tensor_800 est de forme (B, 2, 20, 20)
        state_np = state_tensor_800.cpu().numpy().squeeze(0)  # Convertir en numpy pour la gestion

        total_reward = 0.0
        
        # Le but atteint (reward > 0.5) est notre signal pour changer la cible locale.
        # L'objectif est de remplacer self.goal par une NOUVELLE cible locale menant à FINAL_GOAL_TENSOR.

        for step in range(Params.MAX_TEST_STEPS):
            prev_reward = -1
            
            # --- 2. Mise à jour Dynamique de la Cible Locale (Simuler le Planificateur de Haut Niveau) ---
            
            # Distance à la cible globale (calculé en dehors de l'environnement pour le critère d'arrêt)
            current_pos = env.agent.state.pos.squeeze(0)
            dist_to_final_goal = torch.linalg.norm(current_pos - FINAL_GOAL_TENSOR)

            if dist_to_final_goal < 0.2:  # Cible globale finale atteinte
                total_reward += 10.0  # Récompense bonus pour la fin du parcours
                break 

            # Simuler l'atteinte d'une cible locale (la récompense de +1)
            # Votre environnement renvoie un reward de +1 quand la cible locale est atteinte.
            # Si le reward du pas précédent était +1, cela signifie que nous devons définir une NOUVELLE cible locale.
            if step > 0 and prev_reward > 0.5:
                # Le planificateur de haut niveau place une nouvelle cible LOCALE
                # (self.goal) à l'intersection du rayon Lidar (2m) et du chemin vers la cible finale.

                # 1. Calculer le vecteur vers la cible finale
                target_vec = FINAL_GOAL_TENSOR - current_pos

                # 2. Normaliser ce vecteur
                target_norm = target_vec / torch.linalg.norm(target_vec)

                # 3. La nouvelle cible locale est à la limite du rayon Lidar (2.0m) dans cette direction
                new_local_goal_pos = current_pos + target_norm * env.lidar_range

                # 4. Déplacer l'entité self.goal à cette nouvelle position locale
                env.goal.set_pos(new_local_goal_pos)
            
            # --- 3. Sélection de l'Action (Exploitation Pure) ---
            
            state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
            q_values = q_network(state_tensor)
            action = torch.argmax(q_values).item()

            # --- 4. Exécuter l'action ---
            next_state_tensor_800, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Mettre à jour les états et la récompense précédente
            state_np = next_state_tensor_800.cpu().numpy().squeeze(0)
            prev_reward = reward
            
            # 5. Condition de collision (terminer l'épisode si le robot heurte un obstacle)
            if reward < -0.5:  # Collision: -1 - 0.01 = -1.01
                break
                
        rewards_per_eval_episode.append(total_reward)

    env.training = True  # Revenir au mode entraînement
    average_reward = np.mean(rewards_per_eval_episode)
    return rewards_per_eval_episode, average_reward

def update_L(n_steps):
    """
    Met à jour la distance L entre le point de départ et la cible selon l'Équation (10) [cite: 135-139].
    """
    if n_steps <= Params.N1_THRESHOLD:
        L = Params.L_MIN
    elif Params.N1_THRESHOLD < n_steps < Params.N2_THRESHOLD:
        L = Params.L_MIN + Params.M_SEARCH_SPEED * (n_steps - Params.N1_THRESHOLD)
    else:
        L = Params.L_MAX
    
    return L
