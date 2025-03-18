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



def l1_distance(pos, goal_area):
    """Calculate Manhattan distance to the closest point in goal area"""
    if isinstance(goal_area, list):
        return min(abs(pos[0] - g[0]) + abs(pos[1] - g[1]) for g in goal_area)
    else:
        return abs(pos[0] - goal_area[0]) + abs(pos[1] - goal_area[1])

def compute_reward(
    num_agents,
    old_positions,
    agent_positions,
    evacuated_agents,
    deactivated_agents,
    goal_area,
    com_range

):
    rewards = np.zeros(num_agents)
    new_evacuated_agents = evacuated_agents.copy()
    
    # Global team reward component - will be added to all active agents
    team_reward = 0
    previously_evacuated = len(evacuated_agents)
    currently_deactivated = len(deactivated_agents)
    
    # Parameters - On peut fine tune
    TIME_STEP_PENALTY = -1  # Penalty for each time step (encourages speed)
    NOT_MOVING_PENALTY = -2  # Penalty for staying in the same place
    GOAL_REWARD = 1000.0  # Reward for reaching the goal
    DEACTIVATION_PENALTY = -100.0  # Penalty for agent deactivation
    DISTANCE_REWARD_SCALE = 1.1  # Scale factor for distance-based rewards
    TEAM_SUCCESS_REWARD = 100.0  # Reward for team members when an agent reaches goal
    TEAM_FAILLURE_REWARD = 10.0  # Reward for team members when an agent reaches goal
    EVACUATION_BONUS = 50.0 * num_agents  # Bonus that increases with more agents evacuated
    AGENT_TOO_CLOSE_PENALTY = -3  # Penalty for agents being too close to each other
    AGENT_MIN_DISTANCE = 1  # Minimum desired distance between agents
    AGENT_TOO_FAR_PENALTY = -5  # Penalty for being out of communication range
    COMMUNICATION_RANGE = com_range
    
    # Compute per-agent rewards
    for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
        # Skip already evacuated agents
        if i in evacuated_agents:
            continue
            
        # Handle deactivated agents
        if i in deactivated_agents:
            rewards[i] = DEACTIVATION_PENALTY
            # Penalize the whole team slightly when an agent is deactivated
            team_reward -= TEAM_FAILLURE_REWARD
            continue
            
        # Handle goal achievement
        if tuple(new_pos) in goal_area:
            # Big reward for reaching the goal
            rewards[i] = GOAL_REWARD
            new_evacuated_agents.add(i)
            
            # Team reward when an agent reaches goal
            team_reward += TEAM_SUCCESS_REWARD
            
            # Extra bonus based on how many agents have been evacuated
            team_reward += EVACUATION_BONUS * (len(new_evacuated_agents) / num_agents)
            continue
            
        # Basic movement rewards
        rewards[i] = TIME_STEP_PENALTY  # Base penalty for taking time
        
        # Penalty for not moving
        if np.array_equal(old_pos, new_pos):
            rewards[i] += NOT_MOVING_PENALTY
            
        # Distance-based reward
        old_distance = l1_distance(old_pos, goal_area)
        new_distance = l1_distance(new_pos, goal_area)
        distance_reward = (old_distance - new_distance) * DISTANCE_REWARD_SCALE
        rewards[i] += distance_reward
        connected_agents = 0
        total_agents = 0
        
        for j, other_pos in enumerate(agent_positions):
            if j != i and j not in evacuated_agents and j not in deactivated_agents:
                total_agents += 1
                distance = np.linalg.norm(new_pos - other_pos)
                
                # Penalty for being too close
                if distance < AGENT_MIN_DISTANCE:
                    # Penalty increases as agents get closer
                    proximity_penalty = AGENT_TOO_CLOSE_PENALTY * (1 - distance/AGENT_MIN_DISTANCE)
                    rewards[i] += proximity_penalty
                
                # Track if within communication range
                if distance <= COMMUNICATION_RANGE:
                    connected_agents += 1
                else:
                    # Penalty for being out of communication range
                    # The penalty increases as the agent gets further beyond the communication range
                    out_of_range = distance - COMMUNICATION_RANGE
                    # Cap the penalty to avoid extreme values
                    out_of_range = min(out_of_range, COMMUNICATION_RANGE)
                    communication_penalty = AGENT_TOO_FAR_PENALTY * (out_of_range / COMMUNICATION_RANGE)
                    rewards[i] += communication_penalty
        
        # Connectivity ratio reward - encourages maintaining communication with most agents
        if total_agents > 0:
            connectivity_ratio = connected_agents / total_agents
            # Reward increasing connectivity, with diminishing returns
            # Square root function gives more reward for first connections
            connectivity_reward = 0.3 * np.sqrt(connectivity_ratio)
            rewards[i] += connectivity_reward

    # Add team reward component to all active agents
    for i in range(num_agents):
        if i not in evacuated_agents and i not in deactivated_agents:
            rewards[i] += team_reward / max(1, (num_agents - len(evacuated_agents) - len(deactivated_agents)))
    
    return rewards, new_evacuated_agents
