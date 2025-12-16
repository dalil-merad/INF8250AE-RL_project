class Params:
    """
    Classe contenant les hyperparamètres et configurations pour l'agent RL.
    Basée sur le rapport de projet et les pratiques courantes en RL.
    """
    # --- 1. Hyperparamètres (basés sur le rapport et les pratiques RL) ---
    # Taux d'apprentissage initial de l'optimiseur Adam
    LEARNING_RATE_START = 0.0001
    #Taux de décroissance du learning rate
    LEARNING_DECAY = 0.9999
    # Facteur d'actualisation (Discount factor) [cite: 64]
    GAMMA = 0.99
    # Epsilon initial pour l'exploration [cite: 221]
    EPSILON_START = 1.0
    # Epsilon minimum
    EPSILON_MIN = 0.001
    # Taux de décroissance Epsilon (dépend du nombre total d'étapes)
    EPSILON_DECAY = 0.9995
    # Définition de l'hyperparamètre d'écrêtage (norme maximale)
    MAX_GRADIENT_NORM = 5.0 # À ajuster, 10.0 est une bonne valeur de départ
    # Taux de mise à jour douce du réseau cible
    TAU = 0.001 # Valeur typique entre 0.001 et 0.01

    # Paramètres du Replay Buffer [cite: 262, 266]
    REPLAY_BUFFER_CAPACITY = 80000
    BATCH_SIZE = 64
    TRAINING_START_STEPS = 20000  # Remplissage du buffer avant l'entraînement [cite: 264]
    TRAINING_FREQUENCY_STEPS = 2       # Entraînement toutes les 4 étapes [cite: 265]
    TARGET_UPDATE_FREQUENCY = 400  # Mise à jour du réseau cible tous les C pas

    # Paramètres de l'environnement
    CNN_INPUT_CHANNELS = 2 # 2 canaux (Angle, Distance) pour l'entrée 20x20x2
    ACTION_SIZE = 8        # 8 directions de mouvement [cite: 107]
    NUM_EPISODES = 35000 #40000   # Le rapport utilise 30000-40000 epochs pour la convergence
    MAX_STEPS_PER_EPISODE = 100 # Étapes fixées pour l'entraînement [cite: 163]
    MAX_TEST_STEPS = 1000  # Étapes maximales pour l'évaluation

    # Paramètres du Curriculum Learning (Distance L) [cite: 135-139]
    # Ces valeurs doivent être ajustées selon l'expérience. Ici, une suggestion:
    L_MIN = 0.2  # Distance de départ min (en m)
    L_MAX = 2.0  # Distance de départ max
    N1_THRESHOLD = 1*NUM_EPISODES/8  # Étape 1: Maintient L_MIN
    N2_THRESHOLD = 7*NUM_EPISODES/8 # Étape 2: Atteint L_MAX (n_steps total)
    M_SEARCH_SPEED = (L_MAX - L_MIN) / (N2_THRESHOLD - N1_THRESHOLD)