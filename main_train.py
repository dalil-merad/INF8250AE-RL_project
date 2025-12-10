from environment.path_planning import PathPlanningScenario  # Assurez-vous d'importer la bonne classe
from agent.ddqn_agent import QNetwork, update_L  # evaluate_agent
from agent.params import Params  # Importer tous les hyperparamètres
import torch
import torch.nn as nn
import torch.optim as optim
from vmas import make_env
import collections  # Ajouté pour les types d'actions
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from tqdm import tqdm
import csv
import numpy as np
from utils_eval import generate_plots

CNN_INPUT_CHANNELS = 2
ACTION_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_step = 0
epsilon = Params.EPSILON_START
NUM_ENVS = 32 #1024

# Optionnel: rendu pendant l'entraînement
RENDER_DURING_TRAINING = False      # passez à True pour activer
RENDER_EVERY_N_STEPS = 10           # tous les N steps globaux, on déclenche une fenêtre de rendu
RENDER_NUM_STEPS = 5                # nombre de steps consécutifs à rendre après déclenchement

LOG_EVERY_N_EPISODES = 20  # Fréquence de logging des statistiques

def training_loop():
    global current_step, epsilon
    # Initialisation de l'environnement VMAS (PathPlanningScenario)
    env = make_env(
        scenario=PathPlanningScenario(), 
        num_envs=NUM_ENVS, 
        continuous_actions=False,
        max_steps=Params.MAX_STEPS_PER_EPISODE,
        device=DEVICE, 
        seed=0,
        dict_spaces=True,
        multidiscrete_actions=True,  # <- tell VMAS we use MultiDiscrete
    ) 

    # Initialisation des réseaux (CNN)
    q_network = QNetwork(state_size=CNN_INPUT_CHANNELS, action_size=ACTION_SIZE).to(DEVICE)
    target_q_network = QNetwork(state_size=CNN_INPUT_CHANNELS, action_size=ACTION_SIZE).to(DEVICE)
    target_q_network.load_state_dict(q_network.state_dict())  # Initialisation du réseau cible
    target_q_network.eval()  # Le réseau cible ne doit pas être en mode entraînement

    # Autres initialisations
    optimizer = optim.Adam(q_network.parameters(), lr=Params.LEARNING_RATE)
    criterion = nn.MSELoss()

    # Utiliser le replay buffer de torchrl
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(Params.REPLAY_BUFFER_CAPACITY, device=DEVICE)
    )

    rewards_per_episode = collections.deque(maxlen=100)  # Utiliser deque pour la moyenne glissante

    #Initialisation listes et variables pour analyse des resultats
    results = {}
    cumulative_rewards_avg = []
    len_Reward_episode = 0
    losses = []
    cumulative_loss = 0

    # Compteurs globaux
    total_episodes = 0
    samples_since_last_train = 0
    # Compteur local pour le rendu multi-steps
    render_steps_remaining = 0
    training_step = 0

    max_dist_L = update_L(total_episodes)
    env.scenario.set_max_dist(max_dist_L)

    # Réinitialisation initiale de tous les environnements
    state_dict = env.reset()            # dict d'obs par agent
    state_tensor = state_dict["robot"]  # (NUM_ENVS, 2, 20, 20)
    total_reward = torch.zeros(NUM_ENVS, dtype=torch.float32, device=DEVICE)

    log_f = open("training_log.csv", mode="w", newline="")
    log_writer = csv.writer(log_f)
    log_writer.writerow(["episode", "step", "avg_reward", "loss", "epsilon"])
    current_loss = 0.0
    # --- Boucle d'entraînement principale basée sur le nombre total d'épisodes ---
    pbar = tqdm(total=Params.NUM_EPISODES)
    while total_episodes < Params.NUM_EPISODES:

        # C. Sélection de l'action (Epsilon-greedy) en parallèle pour tous les envs
        with torch.no_grad():
            q_values = q_network(state_tensor)           # (NUM_ENVS, ACTION_SIZE)
            exploitative_actions = torch.argmax(q_values, dim=1)  # (NUM_ENVS,)

            is_explore = torch.rand(NUM_ENVS, device=DEVICE) < epsilon
            random_actions_indices = torch.randint(
                low=0, high=ACTION_SIZE, size=(NUM_ENVS,), device=DEVICE
            )
            action_indices = torch.where(is_explore, random_actions_indices, exploitative_actions)

        # D. Exécution de l'action (parallèle)
        discrete_actions = action_indices.unsqueeze(-1)  # (NUM_ENVS, 1) pour MultiDiscrete
        next_state_dict, reward_dict, done_tensor, info_dict = env.step([discrete_actions])

        # Optionnel: rendu pendant l'entraînement (par ex. seulement env_index=0)
        if RENDER_DURING_TRAINING:
            if render_steps_remaining > 0:
                env.render(env_index=0, mode="human")
                render_steps_remaining -= 1
            elif current_step % RENDER_EVERY_N_STEPS == 0:
                # Démarrer une nouvelle fenêtre de rendu
                render_steps_remaining = RENDER_NUM_STEPS
                env.render(env_index=0, mode="human")
                render_steps_remaining -= 1

        next_state_tensor = next_state_dict["robot"]          # (NUM_ENVS, 2, 20, 20)
        reward_tensor = reward_dict["robot"].to(DEVICE)       # (NUM_ENVS,)
        done_tensor = done_tensor.to(DEVICE)                  # (NUM_ENVS,) bool

        # Mise à jour des récompenses cumulées par environnement
        total_reward += reward_tensor

        # --- Ajout au replay buffer en parallèle pour tous les envs ---
        transition = TensorDict(
            {
                "state": state_tensor,                     # (NUM_ENVS, 2, 20, 20)
                "action": action_indices,                  # (NUM_ENVS,)
                "reward": reward_tensor,                   # (NUM_ENVS,)
                "next_state": next_state_tensor,           # (NUM_ENVS, 2, 20, 20)
                "done": done_tensor.float(),               # (NUM_ENVS,)
            },
            batch_size=[NUM_ENVS],
            device=DEVICE,
        )
        replay_buffer.extend(transition)
        samples_since_last_train += NUM_ENVS
        current_step += NUM_ENVS

        # --- Comptage des épisodes terminés & logging des retours ---
        done_mask = done_tensor.bool()                      # (NUM_ENVS,)
        episodes_this_step = done_mask.long().sum().item()

        if episodes_this_step > 0:
            done_indices = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)

            # Enregistrer les récompenses d'épisode pour les envs terminés
            for idx in done_indices.tolist():
                rewards_per_episode.append(total_reward[idx].item())
                total_reward[idx] = 0.0  # reset cumul pour ce env
                len_Reward_episode += 1
                if len_Reward_episode >= 100:
                    avg_reward_100 = np.mean(rewards_per_episode)
                    cumulative_rewards_avg.append(avg_reward_100)
                    len_Reward_episode = 0

            total_episodes += episodes_this_step
            pbar.update(episodes_this_step)

            # Logging
            if total_episodes % LOG_EVERY_N_EPISODES == 0:
                avg_r = sum(rewards_per_episode) / len(rewards_per_episode) if rewards_per_episode else 0
                print('\r\033[2K\033[1G', end="", flush=True)
                print(
                    f"Ep: {total_episodes} | Avg R: {avg_r:.2f} | Loss: {current_loss:.4f} | Eps: {epsilon:.2f}")

                log_writer.writerow([total_episodes, current_step, avg_r, current_loss, epsilon])
                log_f.flush()

            # --- Reset des environnements terminés ---
            # On réinitialise chaque env terminé, sans récupérer les obs ici
            for idx in done_indices.tolist():
                env.reset_at(
                    idx,
                    return_observations=False,
                    return_info=False,
                    return_dones=False,
                )

            # Après tous les resets, on récupère une seule fois les observations pour tous les envs
            obs_dict = env.get_from_scenario(
                get_observations=True,
                get_rewards=False,
                get_infos=False,
                get_dones=False,
                dict_agent_names=True,
            )
            # get_from_scenario renvoie [obs_dict] quand seuls les obs sont demandés
            if isinstance(obs_dict, list):
                obs_dict = obs_dict[0]
            latest_robot_obs = obs_dict["robot"]        # (NUM_ENVS, 2, 20, 20)

            # Tous les envs ont maintenant dans world l'état correct:
            # - envs done -> reset
            # - envs non done -> états avancés par world.step()
            state_tensor = latest_robot_obs
        else:
            # Aucun env n'est terminé: tous avancent avec next_state_tensor
            state_tensor = next_state_tensor

        # --- Entraînement DDQN: déclenché par le nombre d'échantillons ajoutés ---
        # Fréquence: tous les NUM_ENVS * TRAINING_FREQUENCY_STEPS échantillons ajoutés
        if (
            len(replay_buffer) >= Params.TRAINING_START_STEPS
            and samples_since_last_train/NUM_ENVS % Params.TRAINING_FREQUENCY_STEPS == 0
            # and samples_since_last_train % (NUM_ENVS * TRAINING_FREQUENCY_STEPS) == 0
        ):
            batch = replay_buffer.sample(Params.BATCH_SIZE)  # TensorDict de taille (BATCH_SIZE,)

            states_tensor = batch["state"].to(DEVICE)        # (B, 2, 20, 20)
            actions_tensor = batch["action"].to(DEVICE).long()      # (B,)
            rewards_batch = batch["reward"].to(DEVICE).float()      # (B,)
            next_states_batch = batch["next_state"].to(DEVICE)      # (B, 2, 20, 20)
            dones_batch = batch["done"].to(DEVICE).float()          # (B,)

            # --- DDQN target ---
            with torch.no_grad():
                next_actions = torch.argmax(q_network(next_states_batch), dim=1, keepdim=True)   # (B,1)
                next_q_target = target_q_network(next_states_batch).gather(1, next_actions).squeeze(1)  # (B,)
                target_q_values = rewards_batch + Params.GAMMA * next_q_target * (1.0 - dones_batch)

            predicted_q_values = q_network(states_tensor).gather(
                1, actions_tensor.unsqueeze(1)
            ).squeeze(1)

            loss = criterion(predicted_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            cumulative_loss += current_loss

            # Mise à jour Epsilon (par batch d'entraînement)
            epsilon = max(Params.EPSILON_MIN, epsilon * Params.EPSILON_DECAY)

            # Réinitialiser le compteur d'échantillons depuis le dernier entraînement
            samples_since_last_train = 0
            training_step += 1

        # --- Mise à jour du réseau cible à une fréquence en nombre de steps ---
        if training_step % Params.TARGET_UPDATE_FREQUENCY == 0:
            target_q_network.load_state_dict(q_network.state_dict())
        
        if training_step % 100 == 0:
            losses.append(np.mean(cumulative_loss/100))
            cumulative_loss = 0

        # Curriculum Learning basé sur le nombre total d'épisodes terminés
        max_dist_L = update_L(total_episodes)
        env.scenario.set_max_dist(max_dist_L)
    pbar.close()
    log_f.close()
    results["average_reward"] = cumulative_rewards_avg
    results["last_loss"] = losses[0:160]
    results["first_loss"] = losses[550:1200]
    return q_network, results

# [Optionnel: logique d'affichage des récompenses dans rewards_per_episode]

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
        print(f"\nModèle Q-Network sauvegardé avec succès dans: {filename}")
    except Exception as e:
        print(f"\nErreur lors de la sauvegarde du modèle: {e}")

# --- 4. Évaluation Finale ---
# Global Target Pos est la destination finale du parcours test (e.g., [4.0, 1.0])
# GLOBAL_TARGET_POS = np.array([4.0, 1.0]) 
# eval_rewards, avg_reward = evaluate_agent(q_network, env, GLOBAL_TARGET_POS, num_eval_episodes=5)
# print(f"Average reward during evaluation: {avg_reward:.2f}")


if __name__ == "__main__":
    # Sauvegarde de l'agent entraîné
    trained_network, results = training_loop()
    save_agent(trained_network, filename="ddqn_q_network.pt")
    generate_plots(results=results)
