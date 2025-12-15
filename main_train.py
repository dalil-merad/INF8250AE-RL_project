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

import os
import json
import datetime

CNN_INPUT_CHANNELS = 2
ACTION_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_step = 0
epsilon = Params.EPSILON_START
NUM_ENVS = 64  #1024

# Optionnel: rendu pendant l'entraînement
RENDER_DURING_TRAINING = False       # passez à True pour activer
RENDER_EVERY_N_STEPS = 10           # tous les N steps globaux, on déclenche une fenêtre de rendu
RENDER_NUM_STEPS = 5                # nombre de steps consécutifs à rendre après déclenchement

LOG_EVERY_N_EPISODES = 20  # Fréquence de logging des statistiques

# --- DEBUG: action -> delta position mapping ---
DEBUG_ACTION_DELTAS = False          # set False to disable
DEBUG_NUM_ENVS_TO_TRACK = 5
DEBUG_PRINT_EVERY_ENV_STEPS = 10    # prints every N env-steps (i.e., per while-loop iteration)

def _get_robot_pos_from_info(info_dict: dict, device: torch.device) -> torch.Tensor | None:
    """
    Returns positions as (NUM_ENVS, 2) tensor if available, else None.
    Assumes info_dict["robot"] contains per-env positions (as used in utils_eval.path_agent()).
    """
    if not isinstance(info_dict, dict) or "robot" not in info_dict:
        return None
    pos = info_dict["robot"]
    if not torch.is_tensor(pos):
        try:
            pos = torch.as_tensor(pos, device=device)
        except Exception:
            return None
    # tolerate shapes like (NUM_ENVS, 2) or (NUM_ENVS, k>=2)
    if pos.ndim == 1:
        return None
    return pos[..., :2].to(device)

def training_loop(resume_from: str | None = None, output_dir: str | None = None, run_name: str | None = None):
    global current_step, epsilon
    if not output_dir or not run_name:
        raise ValueError("training_loop requires output_dir and run_name (e.g., results/<yy-mm-dd_HH:MM:SS>/).")

    # Initialisation de l'environnement VMAS (PathPlanningScenario)
    env = make_env(
        scenario=PathPlanningScenario(), 
        num_envs=NUM_ENVS, 
        continuous_actions=False,
        max_steps=Params.MAX_STEPS_PER_EPISODE,
        device=DEVICE, 
        #seed=0,
        dict_spaces=True,
        # NOTE:
        # - True  => action space is MultiDiscrete([8]) and agent.action.u[:,0] is (semantically) an index in {0..7}.
        #           In that case PathPlanningScenario.process_action() should NOT remap from [-1,1] -> {0..7}.
        # - False => action space is exposed as a single Discrete(8)-like action; depending on wrapper plumbing,
        #           agent.action.u may arrive as a continuous proxy or as an index. Verify u range before remapping.
        multidiscrete_actions=True,
    ) 

    # Initialisation des réseaux (CNN)
    q_network = QNetwork(state_size=CNN_INPUT_CHANNELS, action_size=ACTION_SIZE).to(DEVICE)
    target_q_network = QNetwork(state_size=CNN_INPUT_CHANNELS, action_size=ACTION_SIZE).to(DEVICE)

    # --- NEW: optionally resume from saved weights ---
    if resume_from is not None:
        state_dict = torch.load(resume_from, map_location=DEVICE)
        q_network.load_state_dict(state_dict)
        target_q_network.load_state_dict(state_dict)
    else:
        target_q_network.load_state_dict(q_network.state_dict())  # Initialisation du réseau cible

    target_q_network.eval()  # Le réseau cible ne doit pas être en mode entraînement

    # Autres initialisations
    learning_rate = Params.LEARNING_RATE_START
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=Params.LEARNING_DECAY)

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
    # Compteur local pour le rendu multi-steps
    render_steps_remaining = 0
    training_step = 0

    # init per-env episode step counters + termination-type totals (used later)
    episode_steps = torch.zeros(NUM_ENVS, dtype=torch.int32, device=DEVICE)
    total_terminated = 0
    total_truncated = 0

    max_dist_L = update_L(total_episodes)
    env.scenario.set_max_dist(max_dist_L)

    # Réinitialisation initiale de tous les environnements
    state_dict, info_dict = env.reset(return_info=True)            # dict d'obs par agent
    state_tensor = state_dict["robot"]  # (NUM_ENVS, 2, 20, 20)
    total_reward = torch.zeros(NUM_ENVS, dtype=torch.float32, device=DEVICE)

    # --- logging + checkpoints inside the run folder ---
    log_path = os.path.join(output_dir, "training_log.csv")
    ckpt_path = os.path.join(output_dir, f"{run_name}.pt")

    log_f = open(log_path, mode="w", newline="")
    log_writer = csv.writer(log_f)
    log_writer.writerow([
        "episode", "step", "avg_reward", "loss", "epsilon", "L_spawn", "train_step",
        "terminated_this_log", "truncated_this_log", "terminated_total", "truncated_total",
    ])
    current_loss = 0.0
    # --- Boucle d'entraînement principale basée sur le nombre total d'épisodes ---
    pbar = tqdm(total=Params.NUM_EPISODES)
    # --- DEBUG state (track 5 envs) ---
    debug_env_ids = torch.arange(min(DEBUG_NUM_ENVS_TO_TRACK, NUM_ENVS), device=DEVICE)
    debug_last_pos = None
    env_step_count = 0

    try:
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

            # --- DEBUG: log (action_index, Δpos) for a few envs ---
            if DEBUG_ACTION_DELTAS:
                pos_now_all = _get_robot_pos_from_info(info_dict, DEVICE)
                if pos_now_all is not None:
                    pos_now = pos_now_all.index_select(0, debug_env_ids)
                    if debug_last_pos is not None and (env_step_count % DEBUG_PRINT_EVERY_ENV_STEPS == 0):
                        dpos = (pos_now - debug_last_pos).detach().cpu()
                        a_dbg = action_indices.index_select(0, debug_env_ids).detach().cpu()
                        ids = debug_env_ids.detach().cpu().tolist()
                        print("\n[DEBUG action->Δpos]")
                        for i, env_id in enumerate(ids):
                            dx, dy = float(dpos[i, 0].item()), float(dpos[i, 1].item())
                            print(f"  env={env_id:4d} action={int(a_dbg[i].item())}  dpos=({dx:+.4f},{dy:+.4f})")
                    debug_last_pos = pos_now
            env_step_count += 1

            # Optionnel: rendu pendant l'entraînement (par ex. seulement env_index=0)
            if RENDER_DURING_TRAINING:
                if render_steps_remaining > 0:
                    env.render(env_index=0, mode="human")
                    render_steps_remaining -= 1
                elif render_steps_remaining % RENDER_EVERY_N_STEPS == 0:
                    # Démarrer une nouvelle fenêtre de rendu
                    render_steps_remaining = RENDER_NUM_STEPS
                    env.render(env_index=0, mode="human")
                    render_steps_remaining -= 1

            next_state_tensor = next_state_dict["robot"]          # (NUM_ENVS, 2, 20, 20)
            reward_tensor = reward_dict["robot"].to(DEVICE)       # (NUM_ENVS,)
            done_tensor = done_tensor.to(DEVICE)                  # (NUM_ENVS,) bool

            # episode step accounting (increment after env.step)
            episode_steps += 1

            # Split done into truncated vs terminated.
            # Truncated is inferred as hitting time limit. Everything else is treated as terminated.
            done_mask = done_tensor.bool()
            truncated_mask = done_mask & (episode_steps >= Params.MAX_STEPS_PER_EPISODE)
            terminated_mask = done_mask & (~truncated_mask)

            # Mise à jour des récompenses cumulées par environnement
            total_reward += reward_tensor

            # --- Ajout au replay buffer en parallèle pour tous les envs ---
            # IMPORTANT: clone/detach to avoid env/reuse-related corruption in buffer
            transition = TensorDict(
                {
                    "state": state_tensor.detach().clone(),
                    "action": action_indices.detach().clone(),
                    "reward": reward_tensor.detach().clone(),
                    "next_state": next_state_tensor.detach().clone(),
                    "done": done_mask.float().detach().clone(),               # for completeness/debug
                    "terminated": terminated_mask.float().detach().clone(),   # controls bootstrap stop
                    "truncated": truncated_mask.float().detach().clone(),     # logs/analysis
                },
                batch_size=[NUM_ENVS],
                device=DEVICE,
            )
            replay_buffer.extend(transition)
            current_step += NUM_ENVS

            episodes_this_step = done_mask.long().sum().item()

            if episodes_this_step > 0:
                done_indices = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)

                # count termination types
                term_this_step = int(terminated_mask.long().sum().item())
                trunc_this_step = int(truncated_mask.long().sum().item())
                total_terminated += term_this_step
                total_truncated += trunc_this_step

                # Enregistrer les récompenses d'épisode pour les envs terminés
                for idx in done_indices.tolist():
                    rewards_per_episode.append(total_reward[idx].item())
                    total_reward[idx] = 0.0
                    episode_steps[idx] = 0  # reset step count for that env

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
                    print('\r', end="", flush=True)
                    print(
                        f"Ep: {total_episodes} | Avg R: {avg_r:.2f} | Loss: {current_loss:.4f} "
                        + f"| Eps: {epsilon:.2f} | L: {max_dist_L:.2f} | Tstep:{training_step} "
                        + f"| term:{term_this_step} trunc:{trunc_this_step}"
                    )

                    log_writer.writerow([
                        total_episodes, current_step, avg_r, current_loss, epsilon, max_dist_L, training_step,
                        term_this_step, trunc_this_step, total_terminated, total_truncated,
                    ])
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
            did_train = False
            if (
                len(replay_buffer) >= Params.TRAINING_START_STEPS
                and current_step % (NUM_ENVS * Params.TRAINING_FREQUENCY_STEPS) == 0
            ):
                batch = replay_buffer.sample(Params.BATCH_SIZE)  # TensorDict de taille (BATCH_SIZE,)
                states_tensor = batch["state"].to(DEVICE)        # (B, 2, 20, 20)
                actions_tensor = batch["action"].to(DEVICE).long()      # (B,)
                rewards_batch = batch["reward"].to(DEVICE).float()      # (B,)
                next_states_batch = batch["next_state"].to(DEVICE)      # (B, 2, 20, 20)
                terminated_batch = batch["terminated"].to(DEVICE).float()  # <-- only terminated stops bootstrap

                # --- DDQN target ---
                with torch.no_grad():
                    next_actions = torch.argmax(q_network(next_states_batch), dim=1, keepdim=True)   # (B,1)
                    next_q_target = target_q_network(next_states_batch).gather(1, next_actions).squeeze(1)  # (B,)
                    target_q_values = rewards_batch + Params.GAMMA * next_q_target * (1.0 - terminated_batch)

                predicted_q_values = q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

                loss = criterion(predicted_q_values, target_q_values)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), Params.GRAD_CLIP_NORM)
                optimizer.step()

                current_loss = loss.item()
                cumulative_loss += current_loss
                did_train = True

                # Mise à jour Epsilon (par batch d'entraînement)
                epsilon = max(Params.EPSILON_MIN, epsilon * Params.EPSILON_DECAY)
                
                # Mise à jour Learning rate (par batch d'entraînement)
                scheduler.step()
                # clamp LR to LEARNING_RATE_MIN
                lr = max(Params.LEARNING_RATE_MIN, optimizer.param_groups[0]["lr"])
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                training_step += 1

                # update target ONLY after a real training step, and never at step 0
                if training_step % Params.TARGET_UPDATE_FREQUENCY == 0:
                    target_q_network.load_state_dict(q_network.state_dict())

            # --- Logging des pertes ---
            if training_step % 100 == 0 and did_train:
                losses.append(np.mean(cumulative_loss/100))
                cumulative_loss = 0

            if training_step % 50 == 0:
                torch.save(q_network.state_dict(), ckpt_path)

            # Curriculum Learning basé sur le nombre total d'épisodes terminés
            max_dist_L = update_L(total_episodes)
            env.scenario.set_max_dist(max_dist_L)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted (Ctrl-C). Returning current network for checkpointing...")
    finally:
        pbar.close()
        log_f.close()

    # --- 4. Évaluation Finale ---
    results["average_reward"] = cumulative_rewards_avg
    results["loss"] = losses
    return q_network, results

# [Optionnel: logique d'affichage des récompenses dans rewards_per_episode]

def save_agent(q_network, output_dir: str, run_name: str):
    """
    Sauvegarde les poids (state_dict) du Q-Network dans results/<run_name>/<run_name>.pt
    """
    try:
        ckpt_path = os.path.join(output_dir, f"{run_name}.pt")
        torch.save(q_network.state_dict(), ckpt_path)
        print(f"\nModèle Q-Network sauvegardé avec succès dans: {ckpt_path}")
    except Exception as e:
        print(f"\nErreur lors de la sauvegarde du modèle: {e}")

if __name__ == "__main__":
    # --- create the run folder once (keep same name format) ---
    run_name = datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S")
    output_dir = os.path.join("results", run_name)
    os.makedirs(output_dir, exist_ok=True)

    # --- save hyperparameters used alongside outputs ---
    with open(os.path.join(output_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(Params.as_dict(), f, indent=2, ensure_ascii=False)

    # Sauvegarde de l'agent entraîné
    trained_network, results = training_loop(resume_from=None, output_dir=output_dir, run_name=run_name)
    save_agent(trained_network, output_dir=output_dir, run_name=run_name)
    generate_plots(results=results, output_path=output_dir)
