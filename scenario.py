import json
import numpy as np
import os
import simulate

def generate_scenarios(num_scenarios):
    config_ranges = {
        "grid_size": (10, 30),
        "walls_proportion": (0.1, 0.6),
        "num_dynamic_obstacles": (0, 5),
        "num_agents": (4, 4),  # Fixe
        "communication_range": (5, 10),
        "max_lidar_dist_main": (5, 5),  # Fixe
        "max_lidar_dist_second": (1, 3),
        "max_episodes": (10,10), 
        "max_episode_steps": (50,50),
        "render_mode": ("human","human")
    }

    scenario_list = []

    for i in range(num_scenarios):
        fraction = i / (num_scenarios - 1) 

        scenario = {
            "grid_size": int(np.round(np.interp(fraction, [0, 1], config_ranges["grid_size"]))),
            "walls_proportion": round(np.interp(fraction, [0, 1], config_ranges["walls_proportion"]), 2),
            "num_dynamic_obstacles": int(np.round(np.interp(fraction, [0, 1], config_ranges["num_dynamic_obstacles"]))),
            "num_agents": config_ranges["num_agents"][0],
            "communication_range": int(np.round(np.interp(fraction, [0, 1], config_ranges["communication_range"]))),
            "max_lidar_dist_main": config_ranges["max_lidar_dist_main"][0],
            "max_lidar_dist_second": int(np.round(np.interp(fraction, [0, 1], config_ranges["max_lidar_dist_second"]))), 
            "max_episodes": config_ranges["max_episodes"][0] ,
            "max_episode_steps": config_ranges["max_episode_steps"][0],
            "render_mode": config_ranges["render_mode"][0]
        }

        scenario_list.append(scenario)

    return scenario_list

def train_scenarios():
    scenario_paths = sorted([
        os.path.join("eval_configs", f)
        for f in os.listdir("eval_configs") if f.startswith("config_")
    ])

    all_rewards_per_scenario = []  # Liste pour stocker les récompenses de chaque scénario
    trained_agents = []  # Liste pour stocker les agents entraînés

    for i, config_path in enumerate(scenario_paths, 1):
        print(f"\nEntraînement sur scénario {i}: {config_path}")
        
        # Entraîner avec le chemin du fichier de configuration
        trained_agent, all_rewards = simulate.train(config_path)
        print(f"Moyennes des récompenses obtenues pour le scénario {i} : {np.mean(all_rewards)}")
        
        # Ajouter l'agent et les récompenses du scénario
        trained_agents.append(trained_agent)
        all_rewards_per_scenario.append(all_rewards)

    return trained_agents, all_rewards_per_scenario


def plot_cumulated_rewards(scenarios_rewards: list, interval: int = 100):
    """
    Plot and save the rewards over episodes for multiple scenarios.

    Args:
        scenarios_rewards (list of lists): List of total rewards per episode for each scenario.
        interval (int): Interval between ticks on the x-axis (default is 100).
    """
    plt.figure(figsize=(10, 6))

    # Color list for different scenarios
    colors = ["blue", "green", "red", "orange", "purple"]

    # Pour chaque scénario
    for i, rewards in enumerate(scenarios_rewards):
        # Vérifier si rewards est un nombre unique (cas où il n'y a qu'une seule valeur)
        if isinstance(rewards, np.float64):
            rewards = [rewards]  # Convertir en liste si nécessaire
        
        plt.plot(
            range(1, len(rewards) + 1), rewards, 
            color=colors[i % len(colors)], 
            marker="o", linestyle="-", label=f"Scenario {i+1}"
        )
    
    plt.title("Total Cumulated Rewards per Episode for Multiple Scenarios")
    plt.xlabel("Episodes")

    # Ajuster les ticks de l'axe X tous les 'interval' épisodes
    xticks = range(1, len(rewards) + 1, interval)
    plt.xticks(xticks)

    plt.ylabel("Cumulated Rewards")
    plt.grid(True)
    plt.legend()
    plt.savefig("reward_curve_multiple_scenarios.png", dpi=300)
    plt.show()
