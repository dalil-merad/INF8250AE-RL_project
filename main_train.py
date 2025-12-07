from environment.path_planning import PathPlanningScenario # Assurez-vous d'importer la bonne classe
from agent.ddqn_agent import QNetwork, ReplayBuffer, update_L, evaluate_agent
from agent.params import * # Importer tous les hyperparamètres
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from vmas import make_env
import collections # Ajouté pour les types d'actions

CNN_INPUT_CHANNELS = 2
ACTION_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_step = 0
epsilon = EPSILON_START
BATCH_DIM_ENV = 1

def map_discrete_to_continuous_vector(action_indices: torch.Tensor, action_map: torch.Tensor) -> torch.Tensor:
    """
    La méthode la plus rapide pour mapper un grand nombre d'indices discrets vers des vecteurs continus
    en utilisant l'accélération GPU.
    """
    # action_map: Tenseur de forme (8, 2)
    # action_indices: Tenseur de forme (B,)
    
    # Résultat: Tenseur de forme (B, 2), où chaque ligne est le vecteur [Vx, Vy] correspondant à l'indice.
    return torch.index_select(action_map, 0, action_indices)

# Initialisation de l'environnement VMAS (PathPlanningScenario)
env = make_env(scenario=PathPlanningScenario(), num_envs=BATCH_DIM_ENV, device=DEVICE, seed=0)

# 1. Calcul des vitesses basales
speed = 0.1 / env.world.dt
diag_speed = speed * 0.707

# 2. Création du Tenseur de Mappage (8, 2)
action_vectors_list = [
    [ 0.0,  speed], [ 0.0, -speed], [-speed, 0.0], [ speed, 0.0], 
    [-diag_speed, diag_speed], [ diag_speed, diag_speed], 
    [-diag_speed, -diag_speed], [ diag_speed, -diag_speed] 
]
ACTION_MAP_TENSOR = torch.tensor(action_vectors_list, dtype=torch.float32, device=DEVICE)




# Initialisation des réseaux (CNN)
q_network = QNetwork(state_size=CNN_INPUT_CHANNELS, action_size=ACTION_SIZE).to(DEVICE)
target_q_network = QNetwork(state_size=CNN_INPUT_CHANNELS, action_size=ACTION_SIZE).to(DEVICE)
target_q_network.load_state_dict(q_network.state_dict()) # Initialisation du réseau cible
target_q_network.eval() # Le réseau cible ne doit pas être en mode entraînement

# Autres initialisations
optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()
replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
rewards_per_episode = collections.deque(maxlen=100) # Utiliser deque pour la moyenne glissante


# --- 3. Boucle d'Entraînement Complète (Logique DDQN) ---
for episode in range(NUM_EPISODES):
    # A. Curriculum Learning: Mettre à jour L et l'appliquer à l'environnement
    max_dist_L = update_L(episode)
    env.scenario.set_max_dist(max_dist_L)

    # B. Réinitialisation de l'environnement (Map 0 ou 1)
    #state_tensor est de forme (num_envs, num_agents, C, H, W) -> (1, 1, 2, 20, 20)
    state_tensor = env.reset()[0]
    total_reward = torch.zeros(BATCH_DIM_ENV, dtype=torch.float32, device='cpu')

    # C. Sélection de l'action (Epsilon-greedy)
    with torch.no_grad():
        q_values = q_network(state_tensor)
    
        # 1. Exploitation: Meilleure action pour chaque environnement (B,)
        exploitative_actions = torch.argmax(q_values, dim=1) 

        # 2. Exploration: Générer B masques aléatoires pour l'exploration
        is_explore = torch.rand(BATCH_DIM_ENV, device=DEVICE) < epsilon

        # 3. Générer B indices aléatoires (0 à 7) directement
        # Nous utilisons torch.randint, qui produit un tenseur d'indices entiers (Long)
        random_actions_indices = torch.randint(low=0, high=ACTION_SIZE,size=(BATCH_DIM_ENV,),device=DEVICE)
        # 4. Utiliser torch.where pour combiner les actions
        action_indices = torch.where(is_explore,random_actions_indices,exploitative_actions)

    # D. Exécuter l'action et stocker (B>1)
    
    # 1. Format d'Action: VMAS attend List[Tensor]
    # action_indices (B,) -> actions_tensors (B, 1, 1)

    continuous_actions_tensor = map_discrete_to_continuous_vector(action_indices, ACTION_MAP_TENSOR) # retourne (B, 2)
    vmas_actions = [continuous_actions_tensor] 

    # 3. Exécution de l'action
    next_state_tensor, reward_tensor, terminated_tensor, truncated_tensor = env.step(vmas_actions)

    # 4. Mise à jour de l'état/stockage par boucle
    if BATCH_DIM_ENV > 1: # Shape (B, 2, 20, 20)
        next_state_tensor = next_state_tensor.squeeze(1) 
        reward_tensor = reward_tensor.squeeze(1) 
        terminated_tensor = terminated_tensor.squeeze(1) 
        truncated_tensor = truncated_tensor.squeeze(1)
    else:
        next_state_tensor = next_state_tensor[0]
        terminated_tensor = terminated_tensor[0]
        reward_tensor_clean = reward_tensor[0]
        truncated_tensor = truncated_tensor[0]
    # Boucle pour stocker individuellement et gérer les resets
    total_reward_accumulator = total_reward.clone()

    for b in range(BATCH_DIM_ENV):
        # Les tenseurs (clean) sont déjà de la forme la plus propre (B, ...) ou scalaire.

        # a. Extraction de l'état individuel (unbatched)
        # state_tensor est (B, 2, 20, 20). state_tensor[b] -> (2, 20, 20)
        state_b = state_tensor[b].cpu().numpy()
        next_state_b = next_state_tensor[b].cpu().numpy()

        # b. Extraction des scalaires
        # Si B=1, la forme est scalaire. Si B>1, on indexe b.
        if BATCH_DIM_ENV == 1:
            reward_b = reward_tensor_clean.item()
            action_b = action_indices.item()
            # Le terminated est scalaire (Shape [] -> item())
            terminated_b = terminated_tensor.item()
            truncated_b = bool(truncated_tensor) # Conversion de {} en False, ou dict non vide en True
        else:
            reward_b = reward_tensor_clean[b].item()
            action_b = action_indices[b].item()
            terminated_b = terminated_tensor[b].item()
            # Truncated est plus délicat. Nous assumons qu'il est False s'il est un dict vide {}.
            # Si c'était une liste de dicts [{}, {}], il faudrait itérer sur info[b].
            # Par sécurité, nous le gérons comme un booléen simple pour la boucle:
            truncated_b = False # A revoir si B>1 et le format de truncated_tensor change.

        # done_b = True si terminated OU truncated (Standard Gymnasium)
        done_b = terminated_b or truncated_b
    
        # b. Stockage
        replay_buffer.add(state_b, action_b, reward_b, next_state_b, done_b)
    
        # c. Accumulation de récompense
        total_reward_accumulator[b] += reward_b
    
        # d. Gestion des environnements terminés (si done ou collision)
        if done_b or reward_b < -0.5:
            # Si un environnement se termine, il doit être réinitialisé
            # VMAS gère le reset interne, mais nous devons enregistrer le total et 
            # stocker les nouveaux états initiaux pour l'étape suivante.
        
            # Enregistrer la récompense totale pour cet épisode terminé
            rewards_per_episode.append(total_reward_accumulator[b].item())
        
            # Réinitialiser cet environnement B et obtenir le nouvel état initial (s_0)
            # Note: Cette méthode nécessite que votre env supporte reset_at(b) pour la réinitialisation individuelle
            new_obs = env.reset_at(b) 
        
            # Mettre à jour l'état batché et la récompense accumulée
            # Recommencer l'accumulation pour cet env
            total_reward_accumulator[b] = 0.0
        
            # Remplacer l'observation dans le batch global (state_tensor)
            state_tensor[b] = new_obs[0] 

       

        # E. Entraînement par lot (DDQN)
        if len(replay_buffer) > BATCH_SIZE and current_step % TRAINING_FREQUENCY == 0:
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

            # Conversion du lot en Tenseurs (Shape: BATCH_SIZE, 2, 20, 20)
            states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE)
            actions_tensor = torch.tensor(actions, dtype=torch.long).to(DEVICE)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
            next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE)
            #TODO Vérifier les valeurs de dones
            dones = tuple([ 1*(d!=[{}]) for d in dones])
            dones_tensor = torch.tensor(dones, dtype=torch.int).to(DEVICE)

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

            # Mise à jour du compteur de pas
            current_step += 1 # Mettre à jour l'état pour la prochaine itération

        state_np = state_tensor # Maintenant un tensor (B, 1, 2, 20, 20)
            
    # G. Mise à jour du réseau cible
    if (episode + 1) % TARGET_UPDATE_FREQUENCY == 0:
        target_q_network.load_state_dict(q_network.state_dict())

    rewards_per_episode.append(total_reward)
    
    # [Ajouter ici la logique d'affichage des récompenses]

def save_agent(q_network, filename="ddqn_q_network.pt"):
    """
    Sauvegarde les poids (state_dict) du Q-Network dans un fichier .pt.
    
    Args:
        q_network (torch.nn.Module): Le réseau à sauvegarder (le réseau d'apprentissage).
        filename (str): Le nom du fichier de sortie.
    """
    try:
        # On sauvegarde uniquement l'état du modèle (les poids)
        torch.save(q_network.state_dict(), filename)
        print(f"\n✅ Modèle Q-Network sauvegardé avec succès dans: {filename}")
    except Exception as e:
        print(f"\n❌ Erreur lors de la sauvegarde du modèle: {e}")
    
# --- 4. Évaluation Finale ---
# Global Target Pos est la destination finale du parcours test (e.g., [4.0, 1.0])
# GLOBAL_TARGET_POS = np.array([4.0, 1.0]) 
# eval_rewards, avg_reward = evaluate_agent(q_network, env, GLOBAL_TARGET_POS, num_eval_episodes=5)
# print(f"Average reward during evaluation: {avg_reward:.2f}")


# Faire une commande qui sauvegarde le q_network .pt
# remplacer le replay_buffer par celui intégrer celui 




    #state_np = state_tensor[0]# Forme: (B, 2, 20, 20)
    #
    #total_reward = 0
    #
    #for step in range(MAX_STEPS_PER_EPISODE):
    #    current_step += 1
    #    
    #    # C. Sélection de l'action (Epsilon-greedy)
    #    if random.random() < epsilon:
    #        action = env.action_space.sample()
    #    else:
    #        with torch.no_grad():
    #            # state_np est déjà (2, 20, 20), on ajoute la dimension d'environnements
    #            state_tensor = state_np
    #            q_values = q_network(state_tensor)
    #            action = torch.argmax(q_values).item()
    #    
    #    # D. Exécuter l'action et stocker
    #    #TODO executer cette commande sur le nombre d'environnements 
    #    #TODO action est donné directement sous forme de tensor alors qu'on attend ici une liste/un array de tensor
    #    next_state_tensor_800, reward, terminated, truncated = env.step([action])
    #    done = terminated or truncated 
    #    total_reward += reward[0].item()
    #    next_state_np = next_state_tensor_800[0]#.cpu().numpy().squeeze(0)
    #    
    #    replay_buffer.add(state_np, action, reward, next_state_np, done) # S'assurer que de la ligne 39 à la ligne 64, les vecteurs sont de forme (B, 2, 20, 20) ou B est le nb environnement