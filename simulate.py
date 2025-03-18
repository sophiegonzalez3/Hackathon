import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
import time
from typing import Tuple, Optional, Dict

from env import MazeEnv
from agent import MyAgent


def simulation_config(config_path: str, new_agent: bool = True) -> Tuple[MazeEnv, Optional[MyAgent], Dict]:
    """
    Configure the environment and optionally an agent using a JSON configuration file.

    Args:
        config_path (str): Path to the configuration JSON file.
        new_agent (bool): Whether to initialize the agent. Defaults to True.

    Returns:
        Tuple[MazeEnv, Optional[MyAgent], Dict]: Configured environment, agent (if new), and the configuration dictionary.
    """
    
    # Read config
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # Env configuration
    env = MazeEnv(
        size=config.get('grid_size'),                               # Grid size
        walls_proportion=config.get('walls_proportion'),            # Walls proportion in the grid
        num_dynamic_obstacles=config.get('num_dynamic_obstacles'),  # Number of dynamic obstacles
        num_agents=config.get('num_agents'),                        # Number of agents
        communication_range=config.get('communication_range'),      # Maximum distance for agent communications
        max_lidar_dist_main=config.get('max_lidar_dist_main'),      # Maximum distance for main LIDAR scan
        max_lidar_dist_second=config.get('max_lidar_dist_second'),  # Maximum distance for secondary LIDAR scan
        max_episode_steps=config.get('max_episode_steps'),          # Number of steps before episode termination
        render_mode=config.get('render_mode', None),
        seed=config.get('seed', None)                               # Seed for reproducibility
    )

    # Agent configuration
    agent = MyAgent(num_agents=config.get('num_agents')) if new_agent else None

    return env, agent, config


def plot_cumulated_rewards(rewards: list, interval: int = 100):
    """
    Plot and save the rewards over episodes.

    Args:
        rewards (list): List of total rewards per episode.
        interval (int): Interval between ticks on the x-axis (default is 100).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards)+1), rewards, color='blue', marker='o', linestyle='-')
    plt.title('Total Cumulated Rewards per Episode')
    plt.xlabel('Episodes')
    
    # Adjust x-ticks to display every 'interval' episodes
    xticks = range(1, len(rewards)+1, interval)
    plt.xticks(xticks)
    
    plt.ylabel('Cumulated Rewards')
    plt.grid(True)
    plt.savefig('reward_curve_per_episode.png', dpi=300)
    plt.show()


def train(config_path: str) -> MyAgent:
    """
    Train an agent on the configured environment.

    Args:
        config_path (str): Path to the configuration JSON file.

    Returns:
        MyAgent: The trained agent.
    """

    # Environment and agent configuration
    env, agent, config = simulation_config(config_path)
    max_episodes = config.get('max_episodes')

    # Metrics to follow the performance
    all_rewards = []
    total_reward = 0
    episode_count = 0
    
    # Initial reset of the environment
    state, info = env.reset()
    time.sleep(1)

    try:
        while episode_count < max_episodes:
            # Determine agents actions
            actions = agent.get_action(state)

            # Execution of a simulation step
            state, rewards, terminated, truncated, info = env.step(actions)
            total_reward += np.sum(rewards)

            # Update agent policy
            agent.update_policy(actions, state, rewards)

            # Display of the step information
            print(f"\rEpisode {episode_count + 1}, Step {info['current_step']}, "
                  f"Reward: {total_reward:.2f}, "
                  f"Evacuated: {len(info['evacuated_agents'])}, "
                  f"Deactivated: {len(info['deactivated_agents'])}", end='')
            
            # Pause
            time.sleep(1)
            
            # If the episode is terminated
            if terminated or truncated:
                print("\r")
                episode_count += 1
                all_rewards.append(total_reward)
                total_reward = 0
                
                if episode_count < max_episodes:
                    state, info = env.reset()

    except KeyboardInterrupt:
        print("\nSimulation interrupted by the user")
    
    finally:
        env.close()

    return agent, all_rewards


def evaluate(configs_paths: list, trained_agent: MyAgent, num_episodes: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate a trained agent on multiple configurations, calculate metrics, and visualize results.

    Args:
        config_path (list): List of paths to the configuration JSON files.
        trained_agent (MyAgent): A pre-trained agent to evaluate.
        num_episodes (int): Number of episodes to run for evaluation per configuration. Defaults to 10.

    Returns:
        pd.DataFrame: A DataFrame containing evaluation metrics for each episode and configuration.
    """

    # Evaluation results
    all_results = pd.DataFrame()

    for config_path in configs_paths:
        print(f"\n--- Evaluating Configuration: {config_path} ---")

        # Environment configuration
        env, _, config = simulation_config(config_path, new_agent=False)

        # Metrics to follow the performance
        metrics = []
        total_reward = 0
        episode_count = 0
        
        # Initial reset of the environment
        state, info = env.reset()
        time.sleep(1) 
   
        # Run evaluation for the specified number of episodes
        try:
            while episode_count < num_episodes:
                # Determine agents actions
                actions = trained_agent.get_action(state, evaluation=True)

                # Execution of a simulation step
                state, rewards, terminated, truncated, info = env.step(actions)
                total_reward += np.sum(rewards)

                # Display of the step information
                print(f"\rEpisode {episode_count + 1}/{num_episodes}, Step {info['current_step']}, "
                    f"Reward: {total_reward:.2f}, "
                    f"Evacuated: {len(info['evacuated_agents'])}, "
                    f"Deactivated: {len(info['deactivated_agents'])}", end='')
            
                # Pause
                time.sleep(1)

                # If the episode is terminated
                if terminated or truncated:
                    print("\r")
                    # Save metrics
                    metrics.append({
                        "config_path": config_path,
                        "episode": episode_count + 1,
                        "steps": info['current_step'],
                        "reward": total_reward,
                        "evacuated": len(info['evacuated_agents']),
                        "deactivated": len(info['deactivated_agents'])
                    })

                    episode_count += 1
                    total_reward = 0

                    if episode_count < num_episodes:
                        state, info = env.reset()
        
        except KeyboardInterrupt:
            print("\nSimulation interrupted by the user")
        
        finally:
            env.close()

        # Convert the current configuration's metrics to a DataFrame
        config_results = pd.DataFrame(metrics)
        all_results = pd.concat([all_results, config_results], ignore_index=True)
    
    env.close()

    all_results.to_csv('all_results.csv', index=False)

    return all_results