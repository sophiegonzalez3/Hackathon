import json
import numpy as np
import os

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
