import numpy as np


def l1_distance(pos, goal_area):
    min_x_goal = min(goal_area[0][0], goal_area[1][0])
    min_y_goal = min(goal_area[0][1], goal_area[1][1])
    return abs(min_x_goal - pos[0]) + abs(min_y_goal - pos[1])


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
            # penalty for being slow
            time_step_penalty = -0.1

            # penalty for waiting
            not_moving_penalty = -0.1 if np.allclose(old_pos, new_pos) else 0

            # penalty for not advancing toward goal

            # l1_distance_to_goal = l1_distance(new_pos, goal_area)
            # distance_to_goal_penalty = -l1_distance_to_goal / 100

            old_distance = l1_distance(old_pos, goal_area)
            new_distance = l1_distance(new_pos, goal_area)
            distance_to_goal_penalty = old_distance - new_distance

            rewards[i] = (
                time_step_penalty + not_moving_penalty + distance_to_goal_penalty
            ) / 100

    return rewards, evacuated_agents
