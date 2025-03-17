import numpy as np


def compute_reward(
    num_agents,
    old_positions,
    agent_positions,
    evacuated_agents,
    deactivated_agents,
    goal_area,
):
    rewards = np.zeros(num_agents)

    # Compute reward for each agent
    for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
        if i in evacuated_agents:
            continue
        elif i in deactivated_agents:  # Penalties for each deactivated agent
            rewards[i] = -100.0
        elif (
            tuple(new_pos) in goal_area
        ):  # One-time reward for each agent reaching the goal
            rewards[i] = 1000.0
            evacuated_agents.add(i)
        else:
            # Penalties for not finding the goal
            min_x_goal = min(goal_area[0][0], goal_area[1][0])
            min_y_goal = min(goal_area[0][1], goal_area[1][1])
            distance_to_goal = abs(min_x_goal - new_pos[0]) + abs(
                min_y_goal - new_pos[1]
            )
            rewards[i] = -0.1 - distance_to_goal / 100

    return rewards, evacuated_agents
