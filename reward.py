import heapq
import json
from typing import Dict, List, Set, Tuple

import numpy as np


def a_star(grid, start, goal, walls):
    """
    A* algorithm to find the shortest path from start to goal using only static information.

    Args:
        grid: The environment grid
        start: Starting position (x, y)
        goal: Goal position (x, y)
        walls: Set of wall positions

    Returns:
        path_length: Length of the shortest path, or float('inf') if no path exists
        next_step: Next position to move towards goal, or None if no path exists
        full_path: List of positions from start to goal, or empty list if no path exists
    """
    # Convert start and goal to tuples for hashing
    start = tuple(start)
    goal = tuple(goal)

    # Define heuristic function (Manhattan distance)
    def heuristic(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    # Initialize open and closed sets
    open_set = []  # Priority queue of (f_score, position)
    closed_set = set()

    # Initialize g_score (cost from start to current) and f_score (g_score + heuristic)
    g_score = {start: 0}
    f_score = {start: heuristic(start)}
    came_from = {}

    # Add start node to open set
    heapq.heappush(open_set, (f_score[start], start))

    grid_size = len(grid)

    while open_set:
        # Get node with lowest f_score
        _, current = heapq.heappop(open_set)

        # If reached goal, reconstruct path
        if current == goal:
            # Reconstruct path
            full_path = []
            while current in came_from:
                full_path.append(current)
                current = came_from[current]

            # Reverse path to get start to goal order
            full_path.reverse()

            # Return path length, next step, and full path (on a besoin du full path pour le tracer en overlay dans la window)
            if full_path:
                return len(full_path), full_path[0], [start] + full_path
            return 0, start, [start]

        # Add current node to closed set
        closed_set.add(current)

        # Explore neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            # Check if neighbor is valid - only using static information
            if (
                not (0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size)
                or neighbor in walls
                or neighbor in closed_set
            ):
                continue

            # Calculate tentative g_score
            tentative_g_score = g_score[current] + 1

            # If neighbor not in open set or has lower g_score, update it
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor)

                # Add neighbor to open set if not already there
                if neighbor not in [node[1] for node in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    # If no path found
    return float("inf"), None, []


def compute_reward_star(
    num_agents,
    old_positions,
    agent_positions,
    evacuated_agents,
    deactivated_agents,
    goal_area,
    grid=None,
    walls=None,
    dynamic_obstacles=None,
):
    """
    Improved reward function using A* pathfinding .
    to improve :
    * rendre le logging optionnel
    * pas compute le full path si pas besoin

    Args:
        num_agents: Number of agents
        old_positions: Previous positions of agents
        agent_positions: Current positions of agents
        evacuated_agents: Set of agents that reached the goal
        deactivated_agents: Set of deactivated agents
        goal_area: List of goal positions
        grid: The environment grid (static information)
        walls: Set of wall positions (static information)
        dynamic_obstacles: Not used for A* pathfinding, only included for compatibility

    Returns:
        rewards: List of rewards for each agent
        evacuated_agents: Updated set of evacuated agents
    """
    # print(walls)
    # print(grid)

    with open("reward_log.txt", "a") as log_file:
        log_file.write("\n----- New Reward Computation -----\n")
        log_file.write(f"Number of agents: {num_agents}\n")
        log_file.write(f"Agent positions: {agent_positions}\n")
        log_file.write(f"Evacuated agents: {evacuated_agents}\n")
        log_file.write(f"Deactivated agents: {deactivated_agents}\n")
        log_file.write(f"Goal area: {goal_area}\n")
        log_file.write(f"Grid shape: {grid.shape if grid is not None else 'None'}\n")
        log_file.write(
            f"Number of walls: {len(walls) if walls is not None else 'None'}\n"
        )
        # Sample of walls (first 5 if available)
        if walls and len(walls) > 0:
            sample_walls = list(walls)[: min(5, len(walls))]
            log_file.write(f"Sample walls: {sample_walls}\n")

    rewards = np.zeros(num_agents)

    # Convert to set for fast lookups
    if walls is None:
        walls = set()

    # Compute reward for each agent
    for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
        with open("reward_log.txt", "a") as log_file:
            log_file.write(f"In for loop Agent {i + 1}\n")
        if i in evacuated_agents:
            continue
        elif i in deactivated_agents:
            # Penalties for each deactivated agent
            rewards[i] = -100.0
        elif tuple(new_pos) in goal_area:
            # Large one-time reward for reaching goal
            rewards[i] = 1000.0
            evacuated_agents.add(i)
        else:
            # Base penalty for each step
            rewards[i] = -0.1

            # If we have grid information, use A* to improve reward
            if grid is not None:
                # Compute A* path to closest goal using only static information
                closest_goal = min(
                    goal_area,
                    key=lambda g: abs(new_pos[0] - g[0]) + abs(new_pos[1] - g[1]),
                )
                old_path_length, _, _ = a_star(grid, old_pos, closest_goal, walls)
                new_path_length, _, _ = a_star(grid, new_pos, closest_goal, walls)

                # If we moved closer to the goal according to A*
                if new_path_length < old_path_length:
                    # Reward is proportional to the progress made
                    progress = (old_path_length - new_path_length) / max(
                        1, old_path_length
                    )
                    rewards[i] += 10.0 * progress

                    # Log the path improvement
                    with open("reward_log.txt", "a") as log_file:
                        log_file.write(
                            f"Agent {i + 1}: PROGRESS - Old path: {old_path_length}, New path: {new_path_length}, Reward: +{10.0 * progress:.2f}\n"
                        )
                elif new_path_length > old_path_length:
                    # Penalty for moving away from the goal
                    regress = (new_path_length - old_path_length) / max(
                        1, old_path_length
                    )
                    rewards[i] -= 5.0 * regress
                    # Log the path regression
                    with open("reward_log.txt", "a") as log_file:
                        log_file.write(
                            f"Agent {i + 1}: REGRESS - Old path: {old_path_length}, New path: {new_path_length}, Penalty: -{5.0 * regress:.2f}\n"
                        )
                else:  # on penalise aussi si on reste sur place
                    rewards[i] -= 1
                    # Log the path regression
                    with open("reward_log.txt", "a") as log_file:
                        log_file.write(f"Agent {i + 1}: REMAINNED IN PLACE\n")

                # Additional reward for being close to the goal (j'aime pas voir pour une implementation non lineaire de l'appreciation de la distance au goal?)
                if new_path_length <= 3:
                    rewards[i] += (4 - new_path_length) * 2.0
                    with open("reward_log.txt", "a") as log_file:
                        log_file.write(f"Agent {i + 1}: is close to goal\n")

        with open("reward_log.txt", "a") as log_file:
            log_file.write(f"Agent {i + 1}: Reward : {rewards[i]} \n")

    return rewards, evacuated_agents


def l1_distance(pos, goal_area):
    return abs(goal_area[0][0] - pos[0]) + abs(goal_area[0][1] - pos[1])


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
            time_step_penalty = -1

            # penalty for waiting
            not_moving_penalty = -1 if np.allclose(old_pos, new_pos) else 0

            # penalty for not advancing toward goal

            # l1_distance_to_goal = l1_distance(new_pos, goal_area)
            # distance_to_goal_penalty = -l1_distance_to_goal / 100

            # old_distance = l1_distance(old_pos, goal_area)
            new_distance = l1_distance(new_pos, goal_area)
            distance_to_goal_penalty = -new_distance

            rewards[i] = (
                time_step_penalty + not_moving_penalty + distance_to_goal_penalty
            )

    return rewards, evacuated_agents
