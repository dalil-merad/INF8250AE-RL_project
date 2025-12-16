import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
import pandas as pd

MODEL_FOLDER = "results/251213-224333/"
MODEL_PATH = MODEL_FOLDER + "training_log.csv"


def load_results(file_path):
    """
    Load training results from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing training results.
    Returns:
        pd.DataFrame: DataFrame containing the training results.
    """
    return pd.read_csv(file_path)


def plot_avg_reward(results):
    """
    Plot the average reward over episodes.

    Args:
        results (pd.DataFrame): DataFrame containing training results.
    """
    with matplotlib.style.context("seaborn-v0_8"):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(results['episode'], results['avg_reward'], label='Average Reward', marker='o', markersize=4, linewidth=.2)
        # also plot average of last 100 episodes
        results['rolling_avg'] = results['avg_reward'].rolling(window=100).mean()
        ax.plot(results['episode'], results['rolling_avg'], label='Rolling Average (last 100)', color='k', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Reward')
        ax.set_title('Average Reward over Episodes')
        ax.legend()
        plt.tight_layout()
        fig.savefig(MODEL_FOLDER + "avg_reward_plot.png", dpi=150)
        fig.show()

def plot_loss(results):
    """
    Plot the loss over episodes.

    Args:
        results (pd.DataFrame): DataFrame containing training results.
    """
    with matplotlib.style.context("seaborn-v0_8"):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(results['episode'], results['loss'], label='Loss', marker='o', markersize=4, linewidth=.2)
        # also plot average of last 100 episodes
        results['rolling_loss'] = results['loss'].rolling(window=100).mean()
        ax.plot(results['episode'], results['rolling_loss'], label='Rolling Loss (last 100)', color='k', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title('Loss over Episodes (Log Scale)')
        ax.legend()
        ax.set_yscale("log")
        plt.tight_layout()
        fig.savefig(MODEL_FOLDER + "loss_plot.png", dpi=150)
        fig.show()


if __name__ == '__main__':
    experiment_results = load_results(MODEL_PATH)
    plot_avg_reward(experiment_results)
    plot_loss(experiment_results)
