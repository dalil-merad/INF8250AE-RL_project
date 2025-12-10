import numpy as np
import torch
import sys
import os
import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import heapq # Pour gérer la queue de priorité d'A*
from agent.ddqn_agent import QNetwork
from environment.path_planning import PathPlanningScenario 
from vmas import make_env
from environment.map_layouts import MAP_LAYOUTS

def generate_plots(results):
    """
    :param results: Dict with the results needed for the plots
    """
    output_path = 'results/'+datetime.datetime.now().strftime('%y%m%d-%H%M%S')+'/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if results["average_reward"]:
        average_cumulutativ_reward(results["average_reward"], output_path)
    if results["first_loss"]:
        loss_plot(results["first_loss"], output_path, "first")
    if results["last_loss"]:
        print("last loss")
        print(results["last_loss"])
        loss_plot(results["last_loss"], output_path, "end")
    if results["eval_reward"]:
        eval_reward(results["eval_reward"], output_path)

def eval_reward(rewards, path):
    """
    :param rewards: avg_reward list for eval reward
    :param path: path name for plot saving
    """
    timesteps = np.arange(len(rewards))
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, rewards, marker='o', linestyle='-', color='tab:blue')

    plt.title(f'Average Rewards', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Episode reward', fontsize=14)
    plt.legend(fontsize=12)
    file_name = path + f"episode_reward_eval.png"
    plt.savefig(file_name)



def loss_plot(losses, path, step):
    """    
    :param losses: List of loss for each training step
    :param path: path name for plot saving
    :param step: flag for indicating trainning time moment
    """
    timesteps = np.arange(len(losses))
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, losses, marker='o', linestyle='-', color='tab:blue', label='Loss')
    plt.title(f'Loss curve during {step} trainning)', fontsize=16)
    plt.xlabel('Timestep', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    
    plt.legend(fontsize=12)
    file_name = path + f"loss_{step}.png"
    plt.savefig(file_name)


def average_cumulutativ_reward(cumultative_Rewards, path):
    """
    :param Rewards: list of average cumulative reward on each training step
    :param path: path name for plot saving
    """
    epochs = np.arange(len(cumultative_Rewards)) * 100
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, cumultative_Rewards, marker='o', linestyle='-', color='tab:blue', label='Average cumulative reward')
    plt.title(f'Average cumulative reward curve ({len(cumultative_Rewards)} epochs)', fontsize=16)
    plt.xlabel('Epochs/100', fontsize=14)
    plt.ylabel('Average cumulative reward', fontsize=14)
    
    plt.legend(fontsize=12)
    file_name = path + 'cumulative_reward_curve.png'
    plt.savefig(file_name)

def evaluate_agent(self):
    rewards = []
    for ep in range(self.config.num_episodes_eval):
        episode_reward = 0
        done = False
        state = self.env.reset()
        while not done:
            state_norm = torch.FloatTensor(state /
                                        self.config.high)
            with torch.no_grad():
                q_vals = self.q_network(state_norm)
                action = np.argmax(q_vals.numpy())
            next_state, reward, done, info = self.env.step(action)
            next_state = None if done else next_state
            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)
    avg_reward = np.mean(rewards)
    std_error = np.sqrt(np.var(rewards) / len(rewards))
    print(f'Eval average reward: {avg_reward:04.2f} +/-'
        f' {std_error:04.2f}')
    sys.stdout.flush()
    return avg_reward


# --- ALGORITHME A* (A-STAR) ---

def find_path_a_star(grid, start, goal):
    """
    Trouve le chemin le plus court entre un point de départ et un but 
    en utilisant l'algorithme A*.
    
    Args:
        grid (np.array): La carte numérique. 0:sol, 1:mur, 2:but, 3:objet.
        start (tuple): Coordonnées (row, col) du point de départ.
        goal (tuple): Coordonnées (row, col) du but.

    Returns:
        list: Une liste de tuples (row, col) représentant le chemin, ou None.
    """
    
    # Heuristique : Distance de Manhattan
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    rows, cols = grid.shape
    
    # Initialisation
    # La queue de priorité stocke (f_cost, g_cost, row, col)
    # f_cost = g_cost + h_cost
    priority_queue = [(0, 0, start[0], start[1])] 
    
    # g_cost (coût réel du départ au nœud actuel)
    g_cost = {(r, c): float('inf') for r in range(rows) for c in range(cols)}
    g_cost[start] = 0
    
    # parent (pour reconstruire le chemin)
    parent = {}

    # Mouvements possibles (haut, bas, gauche, droite)
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    while priority_queue:
        # Récupère le nœud avec le plus petit f_cost
        f, g, r, c = heapq.heappop(priority_queue)
        current = (r, c)

        if current == goal:
            # Reconstruction du chemin
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            return path[::-1] # Inverse pour avoir Départ -> Arrivée

        # Exploration des voisins
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)

            # Vérifie les limites et les murs (grid[nr, nc] == 1 est un mur)
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != 1:
                
                # Coût du déplacement : 1 pour un mouvement simple
                new_g_cost = g_cost[current] + 1 

                if new_g_cost < g_cost[neighbor]:
                    # Meilleur chemin trouvé
                    g_cost[neighbor] = new_g_cost
                    f_cost = new_g_cost + heuristic(neighbor, goal)
                    parent[neighbor] = current
                    
                    # Ajout ou mise à jour dans la queue de priorité
                    heapq.heappush(priority_queue, (f_cost, new_g_cost, nr, nc))
                    
    return None # Aucun chemin trouvé

# --- FONCTION DE PLOT MISE À JOUR ---

def plot_map_with_path(map_list, start_coords, goal_coords, agent_path = None):
    """
    Visualise la carte avec le départ, le but et le chemin trouvé.
    """

    # I. Définir le Départ et le But (ligne, colonne)
    # Le but 'S' est à (9, 12) dans la carte ci-dessus.
    # Nous allons choisir un point de départ.
    start_pos = start_coords  # Par exemple, à la ligne 4, colonne 2 (à côté du mur WW)
    goal_pos = goal_coords   # Le But 'S' (ligne 9, colonne 13)

    # II. Création de la grille numérique pour A*
    # Le caractère 'S' dans votre carte représente le but.
    # Nous devons d'abord le convertir en grille numérique pour A*.
    height = len(map_list)
    width = len(map_list[0])
    a_star_grid = np.zeros((height, width), dtype=int)

    for r in range(height):
        for c in range(width):
            char = map_list[r][c]
            if char == 'W' or char == 'U':
                a_star_grid[r, c] = 1 # Mur

    # III. Exécution de l'algorithme A*
    path = find_path_a_star(a_star_grid, start_pos, goal_pos)
    
    # 1. Préparation de la grille
    height = len(map_list)
    width = len(map_list[0])
    grid = np.zeros((height, width), dtype=int)
    
    # Conversion des caractères en valeurs numériques pour la grille
    for r in range(height):
        for c in range(width):
            char = map_list[r][c]
            if char == 'W':
                grid[r, c] = 1 # Mur
            elif char == 'U':
                grid[r, c] = 2
            # On ne change pas S et U pour l'instant pour la heatmap
    
    # 2. Définition des couleurs
    # 0: Sol (Gris clair)
    # 1: Mur (Noir)
    # 2: Objet (Bleu)
    # 3: Départ (Orange)
    # 4: But (Rouge)
    # 5: Chemin (Vert)
    
    colors = ['#EEEEEE', '#000000', '#1E90FF', '#FFA500', '#DC143C', '#228B22'] 
    
    # Création d'une grille pour le tracé (nous utilisons 0-1 pour sol/mur, et des valeurs > 1 pour les points spéciaux)
    plot_grid = np.copy(grid) 
    
    # Ajout du départ et du but dans la grille de tracé
    plot_grid[start_coords] = 3  # Départ
    plot_grid[goal_coords] = 4   # But
    
    # Si un chemin est trouvé, l'ajouter à la grille
    if path:
        # On évite d'écraser le départ et l'arrivée dans la visualisation
        for r, c in path[1:-1]: 
            plot_grid[r, c] = 5 # Chemin (entre départ et but)
        
    # Colormap: Mapping des valeurs (0, 1, 3, 5, 6, 7) aux couleurs
    # Nous ajoutons des couleurs supplémentaires pour Départ, But, et Chemin
    cmap_list = [colors[0], colors[1], colors[2], colors[3], colors[4], colors[5]]
    cmap = ListedColormap(cmap_list)
    
    # Les valeurs uniques à tracer
    norm = plt.Normalize(vmin=0, vmax=5)

    # 3. Tracé
    fig, ax = plt.subplots(figsize=(width/3, height/3)) 
    
    # Ajuste l'affichage pour qu'il ne s'étende pas en dehors de la grille
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect('equal')

    # Utilise imshow
    image = ax.imshow(plot_grid, cmap=cmap, norm=norm)
    if agent_path :
        ax.plot(agent_path[1], agent_path[0], color='purple', linestyle='-.', linewidth=1)
    
    # Paramètres d'affichage (grille et tics)
    # Positionne les tics mineurs entre les cellules (à N.5)
    ax.set_xticks(np.arange(-0.5, width, 1), minor=False)
    ax.set_yticks(np.arange(-0.5, height+.5, 1), minor=False)

    # Trace la grille en utilisant les tics mineurs
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
    
    # Supprime les tics majeurs/labels pour un affichage "carte" propre
    ax.tick_params(which='both', length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Titre
    title = "Visualise map navigation"
    ax.set_title(title)
    
    # Légende (Colorbar)
    cbar_ticks = [0, 1, 2, 3, 4, 5]
    cbar_labels = ['Sol (.)', 'Mur (W)', 'Objet (U)', 'Départ', 'But (S)', 'Chemin']
    
    # Masque les ticks non utilisés (comme 2 et 3)
    cbar = fig.colorbar(image, ax=ax, ticks=cbar_ticks, fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(cbar_labels)
    
    plt.show()

def eval_path_agent(newtork_weight):
    start_pos, goal_pos, robot_path, map_number = path_agent(newtork_weight)
    # Tracé du résultat
    map_data = MAP_LAYOUTS[map_number]
    start_pos = (int(remapper_valeur(-start_pos[1], -1.8, 1.8, 0, 18)), int(remapper_valeur(start_pos[0], -2.8, 2.8, 0, 28)))
    goal_pos = (int(remapper_valeur(-goal_pos[1], -1.8, 1.8, 0, 18)), int(remapper_valeur(goal_pos[0], -2.8, 2.8, 0, 28)))
    plot_map_with_path(map_data, start_pos, goal_pos, robot_path)
    plt.figure()
    plt.plot(robot_path[0], robot_path[1])
    plt.show()

def remapper_valeur(valeur, v_min, v_max, t_min, t_max):
    if v_max == v_min:
        return t_min  # Ou lever une erreur, selon le contexte
    
    ratio_normalise = (valeur - v_min) / (v_max - v_min)
    
    # Étape 2: Appliquer ce ratio à l'étendue de l'intervalle cible et ajouter l'offset
    valeur_remappee = t_min + (t_max - t_min) * ratio_normalise
    
    return valeur_remappee

def path_agent(checkpoint_path):

    #charger le réseau
    CNN_INPUT_CHANNELS = 2
    ACTION_SIZE = 8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = QNetwork(state_size=CNN_INPUT_CHANNELS, action_size=ACTION_SIZE).to(DEVICE)
    weight_dict = torch.load(checkpoint_path)
    q_network.load_state_dict(weight_dict)
    q_network.eval()

    path_X = []
    path_Y = []
    done = False
    
    env = make_env(
        scenario=PathPlanningScenario(), 
        num_envs=1, 
        continuous_actions=False,
        max_steps=100,
        device=DEVICE, 
        training = False,
        dict_spaces=True,
        multidiscrete_actions=False  # <- tell VMAS we use MultiDiscrete
    )

    env.scenario.set_max_dist(2.0)
    state_dict = env.reset_at(0) # dict d'obs par agent
    state_tensor = state_dict["robot"]

    map_number = env.scenario.map_choice
    goal = env.scenario.goal.state.pos[0].tolist()
    start = env.agents[0].state.pos.tolist()
    start = start[0]
    step = 0
    render_steps_remaining = 4
    while not done and step < 100:
            with torch.no_grad():
                q_values = q_network(state_tensor)
                action = torch.argmax(q_values).unsqueeze(-1)
            next_state, _, done, info = env.step([action])
            next_state = None if done else next_state
            if next_state:
                state_tensor = next_state["robot"]

            if True:
                if render_steps_remaining > 0:
                    env.render(env_index=0, mode="human")
                    render_steps_remaining -= 1
                elif render_steps_remaining % 5 == 0:
                    # Démarrer une nouvelle fenêtre de rendu
                    render_steps_remaining = 5
                    env.render(env_index=0, mode="human")
                    render_steps_remaining -= 1

            pos = info["robot"][0].tolist()
            pos_x, pos_y = pos[0], pos[1]
            path_X.append(pos_x)
            path_Y.append(pos_y)
            step += 1

    path = [path_X, path_Y]
    return start, goal, path, map_number


if __name__ == "__main__":
    # --- DONNÉES ET EXÉCUTION ---

    map_data = [
        "............................",
        "............................",
        ".........WWWWWWWWWWWWWW.....",
        ".........W..................",
        "...WW....W..................",
        "...WW....W..................",
        ".........W..................",
        ".........W..................",
        ".........W..................",
        ".........W..S......WWW......",
        ".........W.........WWW......",
        "...................WWW......",
        "...U........................",
        "............................",
        "............................",
        "............................",
        ".........WWWWWWWWWW.........",
        ".........................U..",
        "............................"
    ]

    # 1. Définir le Départ et le But (ligne, colonne)
    # Le but 'S' est à (9, 12) dans la carte ci-dessus.
    # Nous allons choisir un point de départ.
    #start_pos = (4, 20)  # Par exemple, à la ligne 4, colonne 2 (à côté du mur WW)
    #goal_pos = (9, 13)   # Le But 'S' (ligne 9, colonne 13)
    #robot_path = [[4, 3, 2 ,5, 6, 8, 9], [20, 20, 18, 15, 16, 14, 13]]
    # 4. Tracé du résultat
    #plot_map_with_path(map_data, start_pos, goal_pos, robot_path)
    eval_path_agent("ddqn_q_network.pt")

"""
results = {}
cumulative_rewards_avg = []
eval_rewards = []

après début l'épsiode
losses = []
cumulative_rewards_100_epoch = []


losses.append(loss)
cumulative_rewards_100_epoch.append()

A la fin de l'épisode
eval_rewards.append(agent.eval_agent())
if episode%100 == 0:
    cumulative_rewards_avg.append(np.mean(cumulative_rewards_100_epoch))
    cumulative_rewards_100_epoch = []
if episode == 0 :
    results["first_loss"] = losses

à la fin de l'entrainement
results["last_loss"] = losses
results["average_reward"] = cumulative_rewards_avg
results["eval_reward"] = eval_rewards
"""