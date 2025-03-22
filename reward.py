import copy
import numpy as np

TOTAL_SCALING = 1000

DEACTIVATION_PENALTY = 1_000_000
ONE_TIME_GOAL_REWARD = 1_000_000

TIME_PENALTY = 0.1
LAZY_PENALTY = 0.5
DISTANCE_REDUCTION_PENALTY_SCALE = 10
DISTANCE_REDUCTION_REWARD = 1
DISTANCE_PENALTY = 1

TEAM_EVACUATION_PROGRESS_REWARD = 5

PROXIMITY_THRESHOLD = 5.0
PROXIMITY_REWARD = 0.3


def manhattan_distance(x, y):
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    return abs(x - y).sum()


def euclidean_distance(x, y):
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    return np.linalg.norm(x - y)


def distance_to_goal_improvement(old_pos, new_pos, goal_area):
    old_goal_distance = manhattan_distance(old_pos, goal_area[0])
    new_goal_distance = manhattan_distance(new_pos, goal_area[0])
    return new_goal_distance - old_goal_distance


def compute_proximity_cooperation(agent_positions):
    cooperation_rewards = np.zeros(len(agent_positions))

    for i, pos_i in enumerate(agent_positions):
        nearby_agents = 0

        for j, pos_j in enumerate(agent_positions):
            if i == j:
                continue
            distance = euclidean_distance(agent_positions[i], agent_positions[j])

            if distance <= PROXIMITY_THRESHOLD:
                nearby_agents += 1
        cooperation_rewards[i] = PROXIMITY_REWARD * nearby_agents

    return cooperation_rewards


def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area):
    rewards = np.zeros(num_agents)

    active_agents = set(range(num_agents)) - evacuated_agents - deactivated_agents

    prev_evacuated_agents = copy.deepcopy(evacuated_agents)

    # Compute reward for each agent
    for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
        if i in evacuated_agents:
            continue
        elif i in deactivated_agents:   # Penalties for each deactivated agent
            rewards[i] = -DEACTIVATION_PENALTY
        elif tuple(new_pos) in goal_area:   # One-time reward for each agent reaching the goal
            rewards[i] = ONE_TIME_GOAL_REWARD
            evacuated_agents.add(i)
        else:
            rewards[i] = -TIME_PENALTY

            rewards[i] -= DISTANCE_PENALTY * manhattan_distance(new_pos, goal_area[0])

            # distance_delta < 0 if the new distance is better than the old distance
            distance_delta = distance_to_goal_improvement(old_pos, new_pos, goal_area)
            rewards[i] -= DISTANCE_REDUCTION_PENALTY_SCALE * distance_delta

            if distance_delta < 0:
                rewards[i] += DISTANCE_REDUCTION_REWARD

            if np.array_equal(old_pos, new_pos):
                rewards[i] -= LAZY_PENALTY

    rewards += TEAM_EVACUATION_PROGRESS_REWARD * (len(evacuated_agents) - len(prev_evacuated_agents))
    rewards += compute_proximity_cooperation(agent_positions)
    rewards /= TOTAL_SCALING
    return rewards, evacuated_agents