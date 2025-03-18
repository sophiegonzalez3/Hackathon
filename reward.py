import numpy as np

def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area):
    rewards = np.zeros(num_agents)
    
    # Track if any new agents have been evacuated in this step
    newly_evacuated = set()

    # Compute reward for each agent
    for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
        if i in evacuated_agents:
            # Already evacuated agents get zero reward
            rewards[i] = 0.0
            continue
        elif i in deactivated_agents:
            # Strong penalty for deactivation (collision)
            rewards[i] = -100.0
            continue
        elif tuple(new_pos) in goal_area:
            # Large reward for reaching the goal
            rewards[i] = 1000.0
            newly_evacuated.add(i)
            continue
        
        # Default small step penalty
        rewards[i] = -0.1
        
        # Calculate distance to the closest goal
        min_old_dist = min(np.linalg.norm(np.array(old_pos) - np.array(goal)) for goal in goal_area)
        min_new_dist = min(np.linalg.norm(np.array(new_pos) - np.array(goal)) for goal in goal_area)
        
        # Reward for getting closer to goal
        distance_delta = min_old_dist - min_new_dist
        
        # Only reward meaningful progress
        if distance_delta > 0.1:
            rewards[i] += distance_delta * 0.5
        
        # Small penalty for not moving
        if np.array_equal(old_pos, new_pos):
            rewards[i] -= 0.05
    
    # Update the evacuated_agents set with newly evacuated agents
    evacuated_agents = evacuated_agents.union(newly_evacuated)

    return rewards, evacuated_agents