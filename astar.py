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
