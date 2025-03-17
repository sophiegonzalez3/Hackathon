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