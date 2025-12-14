from environment.path_planning import PathPlanningScenario 
from agent.ddqn_agent import QNetwork, update_L 
from agent.params import Params 
import torch
import torch.nn as nn
import torch.optim as optim
from vmas import make_env
import collections 
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from tqdm import tqdm
import numpy as np
import time
import wandb

CNN_INPUT_CHANNELS = 2
ACTION_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_step = 0
epsilon = Params.EPSILON_START
NUM_ENVS = 1024

# Configuration WandB
WANDB_CONFIG = {
    "seed": 0, 
    "num_envs": NUM_ENVS,
    "cnn_input_channels": CNN_INPUT_CHANNELS,
    "action_size": ACTION_SIZE,
    "device": str(DEVICE),
    "gamma": Params.GAMMA,
    "epsilon_start": Params.EPSILON_START,
    "epsilon_decay": Params.EPSILON_DECAY,
    "epsilon_min": Params.EPSILON_MIN,
    "target_update_frequency": Params.TARGET_UPDATE_FREQUENCY,
    "learning_rate_start": Params.LEARNING_RATE_START,
    "learning_rate_min": Params.LEARNING_RATE_MIN,
    "learning_decay": Params.LEARNING_DECAY,
    "replay_buffer_capacity": Params.REPLAY_BUFFER_CAPACITY,
    "batch_size": Params.BATCH_SIZE,
    "training_frequency_steps": Params.TRAINING_FREQUENCY_STEPS,
    "max_steps_per_episode": Params.MAX_STEPS_PER_EPISODE,
    "num_episodes": Params.NUM_EPISODES,
}

# Optionnel: rendu pendant l'entraînement
RENDER_DURING_TRAINING = False 
RENDER_EVERY_N_STEPS = 10 
RENDER_NUM_STEPS = 5 

LOG_EVERY_N_EPISODES = 20 

def training_loop():
    global current_step, epsilon
    
    # Initialisation de l'environnement VMAS (PathPlanningScenario)
    env = make_env(
        scenario=PathPlanningScenario(), 
        num_envs=NUM_ENVS, 
        continuous_actions=False,
        max_steps=Params.MAX_STEPS_PER_EPISODE,
        device=DEVICE, 
        #seed=0,
        dict_spaces=True,
        multidiscrete_actions=True,
    ) 

    # Initialisation des réseaux (CNN)
    q_network = QNetwork(state_size=CNN_INPUT_CHANNELS, action_size=ACTION_SIZE).to(DEVICE)
    target_q_network = QNetwork(state_size=CNN_INPUT_CHANNELS, action_size=ACTION_SIZE).to(DEVICE)
    target_q_network.load_state_dict(q_network.state_dict()) 
    target_q_network.eval() 

    # --- INITIALISATION WANDB et DÉFINITION des AXES X ---
    wandb.init(
        project="DDQN-Path-Planning-VMAS", 
        config=WANDB_CONFIG,
        monitor_gym=False 
    )
    wandb.run.name = f"DDQN_B{NUM_ENVS}_LR{Params.LEARNING_RATE_START}" 

    wandb.define_metric("Steps_Total") 
    wandb.define_metric("Episode/*", step_metric="Steps_Total")
    wandb.define_metric("Stats/*", step_metric="Steps_Total")
    wandb.define_metric("Hyperparameters/*", step_metric="Steps_Total")
    
    wandb.define_metric("Training/Steps")
    wandb.define_metric("Training/*", step_metric="Training/Steps")

    wandb.watch(q_network, log="gradients", log_freq=100)

    # Autres initialisations
    learning_rate = Params.LEARNING_RATE_START
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=Params.LEARNING_DECAY)

    # Utiliser le replay buffer de torchrl
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(Params.REPLAY_BUFFER_CAPACITY, device=DEVICE)
    )

    rewards_per_episode = collections.deque(maxlen=100) 

    # Variables pour l'analyse des résultats (Simplifiées pour WandB)
    results = {}
    losses = [] # Variable conservée uniquement pour l'analyse post-training (optionnel)
    cumulative_loss = 0

    # Compteurs globaux
    total_episodes = 0
    # Compteur local pour le rendu multi-steps
    render_steps_remaining = 0
    training_step = 0

    max_dist_L = update_L(total_episodes)
    env.scenario.set_max_dist(max_dist_L)

    # Réinitialisation initiale de tous les environnements
    state_dict = env.reset() 
    state_tensor = state_dict["robot"] 
    total_reward = torch.zeros(NUM_ENVS, dtype=torch.float32, device=DEVICE)

    # Suppression du système de logging CSV obsolète
    current_loss = 0.0
    
    # --- Boucle d'entraînement principale ---
    pbar = tqdm(total=Params.NUM_EPISODES)
    while total_episodes < Params.NUM_EPISODES:

        # C. Sélection de l'action (Epsilon-greedy)
        with torch.no_grad():
            q_values = q_network(state_tensor) 
            exploitative_actions = torch.argmax(q_values, dim=1) 

            is_explore = torch.rand(NUM_ENVS, device=DEVICE) < epsilon
            random_actions_indices = torch.randint(
                low=0, high=ACTION_SIZE, size=(NUM_ENVS,), device=DEVICE
            )
            action_indices = torch.where(is_explore, random_actions_indices, exploitative_actions)

        # D. Exécution de l'action (parallèle)
        discrete_actions = action_indices.unsqueeze(-1) 
        t_start_env = time.time()
        next_state_dict, reward_dict, done_tensor, info_dict = env.step([discrete_actions])
        torch.cuda.synchronize() # Synchro pour mesurer le temps GPU
        t_end_env = time.time()
        # Suppression du print : print(f"Env.step time: {t_end_env - t_start_env:.4f}s") 

        # Optionnel: rendu pendant l'entraînement
        if RENDER_DURING_TRAINING:
            # ... (logique de rendu) ...
            pass # Rendu non critique pour le logging

        next_state_tensor = next_state_dict["robot"] 
        reward_tensor = reward_dict["robot"].to(DEVICE) 
        done_tensor = done_tensor.to(DEVICE) 

        total_reward += reward_tensor

        # --- Ajout au replay buffer ---
        transition = TensorDict(
            {
                "state": state_tensor, 
                "action": action_indices, 
                "reward": reward_tensor, 
                "next_state": next_state_tensor, 
                "done": done_tensor.float(),
            },
            batch_size=[NUM_ENVS],
            device=DEVICE,
        )
        t_start_extend = time.time()
        replay_buffer.extend(transition)
        torch.cuda.synchronize()
        t_end_extend = time.time()
        # Suppression du print : print(f"Buffer.extend time: {t_end_extend - t_start_extend:.4f}s") 
        
        current_step += NUM_ENVS

        # --- Comptage des épisodes terminés & logging des retours ---
        done_mask = done_tensor.bool() 
        episodes_this_step = done_mask.long().sum().item()

        if episodes_this_step > 0:
            done_indices = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)

            # Enregistrer les récompenses d'épisode pour les envs terminés
            for idx in done_indices.tolist():
                reward_value = total_reward[idx].item()
                rewards_per_episode.append(reward_value) 

                # LOGGING INDIVIDUEL (CRITIQUE pour l'analyse)
                wandb.log(
                {
                    "Episode/Reward_Value": reward_value,
                    "Steps_Total": current_step, 
                    "Total_Episodes": total_episodes, 
                },
                step=current_step
                )
                
                total_reward[idx] = 0.0  # reset cumul pour ce env
                # Suppression de la logique len_Reward_episode obsolète

            total_episodes += episodes_this_step
            pbar.update(episodes_this_step)

            # Logging de la MOYENNE GLISSANTE (DRL That Matters)
            if total_episodes % LOG_EVERY_N_EPISODES == 0 and len(rewards_per_episode) > 0:
                avg_r = sum(rewards_per_episode) / len(rewards_per_episode)
                std_r = np.std(rewards_per_episode)
                
                wandb.log(
                {
                "Stats/Avg_Reward_100_Episodes": avg_r,
                "Stats/Std_Reward_100_Episodes": std_r, 
                "Hyperparameters/Epsilon": epsilon,
                "Hyperparameters/Max_Dist_L": max_dist_L,
                "Training/Loss_Avg_Last": current_loss,
                "Steps_Total": current_step,
                },
                step=current_step 
                )
                
                # Suppression du logging console/CSV obsolète
                print('\r', end="", flush=True)
                print(
                    f"Ep: {total_episodes} | Avg R: {avg_r:.2f} | Std R: {std_r:.2f} | Loss: {current_loss:.4f} " 
                    + f"| Eps: {epsilon:.2f} | L: {max_dist_L:.2f} | Tstep:{training_step}")

            # --- Reset des environnements terminés ---
            for idx in done_indices.tolist():
                env.reset_at(
                    idx,
                    return_observations=False,
                    return_info=False,
                    return_dones=False,
                )

            obs_dict = env.get_from_scenario(
                get_observations=True,
                get_rewards=False,
                get_infos=False,
                get_dones=False,
                dict_agent_names=True,
            )
            if isinstance(obs_dict, list):
                obs_dict = obs_dict[0]
            latest_robot_obs = obs_dict["robot"] 

            state_tensor = latest_robot_obs
        else:
            state_tensor = next_state_tensor

        # --- Entraînement DDQN ---
        if (
            len(replay_buffer) >= Params.TRAINING_START_STEPS
            and current_step % (NUM_ENVS * Params.TRAINING_FREQUENCY_STEPS) == 0
        ):
            t_start_train = time.time()
            batch = replay_buffer.sample(Params.BATCH_SIZE) 
            states_tensor = batch["state"].to(DEVICE) 
            actions_tensor = batch["action"].to(DEVICE).long() 
            rewards_batch = batch["reward"].to(DEVICE).float() 
            next_states_batch = batch["next_state"].to(DEVICE) 
            dones_batch = batch["done"].to(DEVICE).float() 

            with torch.no_grad():
                next_actions = torch.argmax(q_network(next_states_batch), dim=1, keepdim=True) 
                next_q_target = target_q_network(next_states_batch).gather(1, next_actions).squeeze(1) 
                target_q_values = rewards_batch + Params.GAMMA * next_q_target * (1.0 - dones_batch)

            predicted_q_values = q_network(states_tensor).gather(
                1, actions_tensor.unsqueeze(1)
            ).squeeze(1)

            loss = criterion(predicted_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            t_end_train = time.time()
            # Suppression du print : print(f"Training time: {t_end_train - t_start_train:.4f}s")

            current_loss = loss.item()
            cumulative_loss += current_loss
            
            # Logging des métriques d'entraînement
            wandb.log(
            {
                "Training/Loss": current_loss, 
                "Training/Steps": training_step,
                "Hyperparameters/Learning_Rate": learning_rate,
                "Hyperparameters/Epsilon_Decay": epsilon 
            }, 
            step=training_step 
            )

            # Mise à jour Epsilon/LR/Compteurs
            epsilon = max(Params.EPSILON_MIN, epsilon * Params.EPSILON_DECAY)
            scheduler.step()
            learning_rate = optimizer.param_groups[0]['lr']
            training_step += 1

        # --- Mise à jour du réseau cible ---
        if training_step % Params.TARGET_UPDATE_FREQUENCY == 0:
            target_q_network.load_state_dict(q_network.state_dict())
            
        # Suppression de l'ancienne logique de perte/analyse
        # if training_step % 100 == 0:
        #    losses.append(np.mean(cumulative_loss/100))
        #    cumulative_loss = 0

        # Curriculum Learning
        max_dist_L = update_L(total_episodes)
        env.scenario.set_max_dist(max_dist_L)
        
    pbar.close()
    
    # --- FINALISATION WANDB et SAUVEGARDE ---
    final_model_path = "ddqn_q_network_final.pt"
    torch.save(q_network.state_dict(), final_model_path)
    wandb.save(final_model_path) 
    wandb.finish()
    
    # --- 4. Évaluation Finale (Simplifiée) ---
    results["average_reward"] = np.mean(rewards_per_episode) if rewards_per_episode else 0.0 # Retourne la moyenne finale
    # Suppression des références aux listes obsolètes : results["loss"] = losses
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

if __name__ == "__main__":
    # Sauvegarde de l'agent entraîné
    print(f'Cuda available: {torch.cuda.is_available()}')
    trained_network, results = training_loop()
    save_agent(trained_network, filename="ddqn_q_network.pt")


    
