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
        "max_episode_steps": (100,500),
        "render_mode": ("human","human")
    }

    scenario_list = []

    for i in range(num_scenarios):
        fraction = i / (num_scenarios - 1) 

        scenario = {
            "grid_size": int(np.round(np.interp(fraction, [0, 1], config_ranges["grid_size"]))),
            "walls_proportion": round(np.interp(fraction, [0, 1], config_ranges["walls_proportion"]), 2),
            "num_dynamic_obstacles": int(np.round(np.interp(fraction, [0, 1], config_ranges["num_dynamic_obstacles"]))),
            "num_agents": int(np.round(np.interp(fraction, [0, 1], config_ranges["num_agents"]))),
            "communication_range": int(np.round(np.interp(fraction, [0, 1], config_ranges["communication_range"]))),
            "max_lidar_dist_main": int(np.round(np.interp(fraction, [0, 1], config_ranges["max_lidar_dist_main"]))),
            "max_lidar_dist_second": int(np.round(np.interp(fraction, [0, 1], config_ranges["max_lidar_dist_second"]))), 
            "max_episodes": int(np.round(np.interp(fraction, [0, 1], config_ranges["max_episodes"]))), 
            "max_episode_steps": int(np.round(np.interp(fraction, [0, 1], config_ranges["max_episode_steps"]))), 
            "render_mode": config_ranges["render_mode"][0]
        }

        scenario_list.append(scenario)

    return scenario_list

def train_scenarios():
    scenario_paths = sorted([
        os.path.join("eval_configs", f)
        for f in os.listdir("eval_configs") if f.startswith("config_")
    ])

    all_rewards_per_scenario = []
    trained_agents = []

    for i, config_path in enumerate(scenario_paths, 1):
        print(f"\nEntraînement sur scénario {i}: {config_path}")
        
        trained_agent, all_rewards = simulate.train(config_path)
        print(f"Moyennes des récompenses obtenues pour le scénario {i} : {np.mean(all_rewards)}")
        
        trained_agents.append(trained_agent)
        all_rewards_per_scenario.append(all_rewards)

    return trained_agents, all_rewards_per_scenario


def plot_cumulated_rewards(scenarios_rewards: list, interval: int = 100):
    plt.figure(figsize=(10, 6))

    colors = ["blue", "green", "red", "orange", "purple", "yellow", "pink", "black", "white", "gray", "violet", "maroon", "turquoise"]

    for i, rewards in enumerate(scenarios_rewards):
        if isinstance(rewards, np.float64):
            rewards = [rewards]
        
        plt.plot(
            range(1, len(rewards) + 1), rewards, 
            color=colors[i % len(colors)], 
            marker="o", linestyle="-", label=f"Scenario {i+1}"
        )
    
    plt.title("Total Cumulated Rewards per Episode for Multiple Scenarios")
    plt.xlabel("Episodes")

    xticks = range(1, len(rewards) + 1, interval)
    plt.xticks(xticks)

    plt.ylabel("Cumulated Rewards")
    plt.grid(True)
    plt.legend()
    plt.savefig("reward_curve_multiple_scenarios.png", dpi=300)
    plt.show()
