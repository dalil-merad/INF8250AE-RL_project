class Params:
    """
    Classe contenant les hyperparamètres et configurations pour l'agent RL.
    Basée sur le rapport de projet et les pratiques courantes en RL.
    """
    # --- 1. Hyperparamètres (basés sur le rapport et les pratiques RL) ---
    # Taux d'apprentissage initial de l'optimiseur Adam
    LEARNING_RATE_START = 0.0001
    # Taux d'apprentissage initial de l'optimiseur Adam
    LEARNING_RATE_MIN = 0.00001
    #Taux de décroissance du learning rate
    LEARNING_DECAY = 1
    # Facteur d'actualisation (Discount factor) [cite: 64]
    GAMMA = 0.99
    # Epsilon initial pour l'exploration [cite: 221]
    EPSILON_START = 1.0
    # Epsilon minimum
    EPSILON_MIN = 0.01
    # Taux de décroissance Epsilon (dépend du nombre total d'étapes)
    EPSILON_DECAY = 0.9995  

    # Paramètres du Replay Buffer [cite: 262, 266]
    REPLAY_BUFFER_CAPACITY = 40000
    BATCH_SIZE = 32
    TRAINING_START_STEPS = 20000  # Remplissage du buffer avant l'entraînement [cite: 264]
    TRAINING_FREQUENCY_STEPS = 4       # Entraînement toutes les 4 étapes [cite: 265]
    TARGET_UPDATE_FREQUENCY = 400  # Mise à jour du réseau cible tous les C pas

    # Paramètres de l'environnement
    CNN_INPUT_CHANNELS = 2 # 2 canaux (Angle, Distance) pour l'entrée 20x20x2
    ACTION_SIZE = 8        # 8 directions de mouvement [cite: 107]
    NUM_EPISODES = 20000 #40000   # Le rapport utilise 30000-40000 epochs pour la convergence
    MAX_STEPS_PER_EPISODE = 100 # Étapes fixées pour l'entraînement [cite: 163]
    MAX_TEST_STEPS = 1000  # Étapes maximales pour l'évaluation

    # Paramètres du Curriculum Learning (Distance L) [cite: 135-139]
    # Ces valeurs doivent être ajustées selon l'expérience. Ici, une suggestion:
    L_MIN = 0.2  # Distance de départ min (en m)
    L_MAX = 2.0  # Distance de départ max
    N1_THRESHOLD = 2*NUM_EPISODES/8  # Étape 1: Maintient L_MIN
    N2_THRESHOLD = 6*NUM_EPISODES/8 # Étape 2: Atteint L_MAX (n_steps total)
    M_SEARCH_SPEED = (L_MAX - L_MIN) / (N2_THRESHOLD - N1_THRESHOLD)