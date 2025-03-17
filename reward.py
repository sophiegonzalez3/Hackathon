import numpy as np

def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area):
    rewards = np.zeros(num_agents)

    # Compute reward for each agent
    for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
        if i in evacuated_agents:
            continue
        elif i in deactivated_agents:   # Penalties for each deactivated agent
            rewards[i] = -100.0
        elif tuple(new_pos) in goal_area:   # One-time reward for each agent reaching the goal
            rewards[i] = 1000.0
            evacuated_agents.add(i)
        else:
            # Penalties for not finding the goal
            rewards[i] = -0.1

    return rewards, evacuated_agents


import numpy as np
from typing import List, Set, Tuple

def compute_reward(num_agents: int, old_positions: list, 
                   new_positions: list, evacuated_agents: Set[int], 
                   deactivated_agents: Set[int], goal_area: List[Tuple[int, int]]) -> float:
    """
    Compute reward for agents based on their actions and new states.
    
    Args:
        num_agents: Number of agents in the environment
        old_positions: Previous positions of all agents
        new_positions: Current positions of all agents
        evacuated_agents: Set of agents that have reached the goal
        deactivated_agents: Set of agents that have been deactivated
        goal_area: List of goal positions for each agent
        
    Returns:
        reward: Team reward value
    """
    # Base reward to discourage long episodes
    base_reward = -0.1
    
    # Initialize rewards
    reward = base_reward
    
    # Check each agent's status and movement
    for i in range(num_agents):
        # Skip already evacuated agents
        if i in evacuated_agents:
            continue
            
        # Heavy penalty for deactivated agents (hitting obstacles or walls)
        if i in deactivated_agents and i not in evacuated_agents:
            reward -= 10.0
            continue
            
        # Convert position arrays to tuples for comparison
        curr_pos = tuple(new_positions[i])
        prev_pos = tuple(old_positions[i])
        goal_pos = goal_area[i]
        
        # Calculate distances
        prev_dist_to_goal = np.sqrt((prev_pos[0] - goal_pos[0])**2 + (prev_pos[1] - goal_pos[1])**2)
        curr_dist_to_goal = np.sqrt((curr_pos[0] - goal_pos[0])**2 + (curr_pos[1] - goal_pos[1])**2)
        
        # Reward for moving closer to the goal
        dist_diff = prev_dist_to_goal - curr_dist_to_goal
        reward += dist_diff * 0.5  # Scale the reward based on how much closer the agent got
        
        # Check if agent has reached the goal
        if curr_pos == goal_pos or (abs(curr_pos[0] - goal_pos[0]) <= 1 and abs(curr_pos[1] - goal_pos[1]) <= 1):
            # Large reward for reaching the goal
            reward += 50.0
            evacuated_agents.add(i)
            
    # Additional team reward for cooperative behavior
    # Calculate average distance between all active agents
    active_agents = set(range(num_agents)) - evacuated_agents - deactivated_agents
    if len(active_agents) > 1:
        total_distance = 0
        num_pairs = 0
        
        for i in active_agents:
            for j in active_agents:
                if i < j:  # Avoid counting pairs twice
                    pos_i = np.array(new_positions[i])
                    pos_j = np.array(new_positions[j])
                    distance = np.linalg.norm(pos_i - pos_j)
                    
                    # Encourage some dispersion (not too close, not too far)
                    # Ideal distance depends on the environment size and number of agents
                    ideal_distance = 5.0  # Adjust based on your environment
                    
                    dispersion_reward = max(0, 1.0 - abs(distance - ideal_distance) / ideal_distance) * 0.1
                    reward += dispersion_reward
                    
                    total_distance += distance
                    num_pairs += 1
    
    # Bonus for having all agents reach the goal
    if len(evacuated_agents) == num_agents:
        reward += 100.0
        
    return reward