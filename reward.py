import numpy as np

def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area):
    rewards = np.zeros(num_agents)
    
    # Compute reward for each agent
    for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
        if i in evacuated_agents:
            continue
        elif i in deactivated_agents:
            # Higher penalty for deactivated agents
            rewards[i] = -100.0
        elif tuple(new_pos) in goal_area:
            # Higher reward for reaching goal
            rewards[i] = 1000.0
            evacuated_agents.add(i)
        else:
            # Base penalty for time spent
            rewards[i] = -0.1
            
            # Calculate distance to closest goal position
            min_distance_to_goal = float('inf')
            for goal_pos in goal_area:
                distance = np.sqrt((new_pos[0] - goal_pos[0])**2 + (new_pos[1] - goal_pos[1])**2)
                min_distance_to_goal = min(min_distance_to_goal, distance)
            
            # Compare with previous distance
            if old_pos is not None and old_pos[0] >= 0 and old_pos[1] >= 0:  # Make sure old position is valid
                old_min_distance = float('inf')
                for goal_pos in goal_area:
                    old_distance = np.sqrt((old_pos[0] - goal_pos[0])**2 + (old_pos[1] - goal_pos[1])**2)
                    old_min_distance = min(old_min_distance, old_distance)
                
                # Reward for moving closer to goal
                if min_distance_to_goal < old_min_distance:
                    rewards[i] += 0.3
                elif min_distance_to_goal > old_min_distance:
                    rewards[i] -= 0.2
            
            # Penalty for staying in the same place
            if np.array_equal(old_pos, new_pos):
                rewards[i] -= 0.1
    
    # Bonus for first agent reaching goal (to encourage leader behavior)
    if len(evacuated_agents) == 1:
        leader_id = next(iter(evacuated_agents))
        rewards[leader_id] += 500.0
    
    # Group reward when all agents reach goal
    if len(evacuated_agents) == num_agents:
        for i in range(num_agents):
            rewards[i] += 2000.0
    
    return rewards, evacuated_agents