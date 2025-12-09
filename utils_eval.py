import numpy as np
import torch
import sys
import os
import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import heapq # Pour gérer la queue de priorité d'A*

def generate_plot(results):
    """
    :param results: Dict with the results needed for the plots
    """
    output_path = 'results/'+datetime.datetime.now().strftime('%y%m%d-%H%M%S')+'/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    average_cumulutativ_reward(results["average_reward"], output_path)
    loss_plot(results["first_loss"], output_path, "first")
    loss_plot(results["last_loss"], output_path, "end")
    eval_reward(results["eval_reward"], output_path)

def eval_reward(rewards, path):
    """
    :param rewards: avg_reward list for eval reward
    :param path: path name for plot saving
    """
    timesteps = len(rewards)
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, rewards, marker='o', linestyle='-', color='tab:blue')

    plt.title(f'Average Rewards', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Episode reward', fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(timesteps)
    file_name = path + f"episode_reward_eval.png"
    plt.savefig(file_name)



def loss_plot(losses, path, step):
    """    
    :param losses: List of loss for each training step
    :param path: path name for plot saving
    :param step: flag for indicating trainning time moment
    """
    timesteps = len(losses)
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, losses, marker='o', linestyle='-', color='tab:blue', label='Loss')
    plt.title(f'Loss curve during {step} trainning)', fontsize=16)
    plt.xlabel('Timestep', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.xticks(timesteps)
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
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.xticks(epochs)
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
            char = map_data[r][c]
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
    print(np.arange(-0.5, height+0.5, 1))

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
    start_pos = (4, 20)  # Par exemple, à la ligne 4, colonne 2 (à côté du mur WW)
    goal_pos = (9, 13)   # Le But 'S' (ligne 9, colonne 13)
    robot_path = [[4, 3, 2 ,5, 6, 8, 9], [20, 20, 18, 15, 16, 14, 13]]
    # 4. Tracé du résultat
    plot_map_with_path(map_data, start_pos, goal_pos, robot_path)

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