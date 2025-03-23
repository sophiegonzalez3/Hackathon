import heapq
from collections import defaultdict

import numpy as np


class MyAgent:
    def __init__(self, num_agents=4):
        """Initialize agents with individual memory maps and A* navigation."""
        self.num_agents = num_agents
        # Track agents states for memory
        self.agent_memories = [None] * num_agents
        # Store the last known grid size
        self.grid_size = 30  # Default size, will update when getting first state
        # Store the goal positions for each agent
        self.goal_positions = [None] * num_agents
        # Store the communication range
        self.communication_range = 8  # Default, will update from observations
        # Store the lidar ranges
        self.max_lidar_dist_main = 5  # Default
        self.max_lidar_dist_second = 3  # Default
        # Store the path for each agent
        self.agent_paths = [[] for _ in range(num_agents)]
        # Store the previous positions to detect if we're stuck
        self.previous_positions = [None] * num_agents
        # Counter for being stuck in the same position
        self.stuck_counter = [0] * num_agents
        # Last actions taken
        self.last_actions = [0] * num_agents
        # Number of times the same action was repeated
        self.action_repeat_counter = [0] * num_agents
        # Maximum repeat count before trying a random action
        self.max_repeat_count = 5
        # Memory of dangerous areas (dynamic obstacles)
        self.danger_zones = [{} for _ in range(num_agents)]
        # Initialize obstacle memory (0: unknown, 1: wall, 2: dynamic obstacle, 3: other agent)
        self.initialize_memories()
        # Add a step counter to track time
        self.time_step = 0
        # Maximum age for danger zones (in steps)
        self.danger_zone_expiry = 3

    def initialize_memories(self):
        """Initialize the memory for all agents."""
        for i in range(self.num_agents):
            # Create a grid of unknowns (0)
            # -1: unknown, 0: free space, 1: wall, 2: dynamic obstacle, 3: other agent, 4: goal
            self.agent_memories[i] = np.ones((self.grid_size, self.grid_size)) * -1

    def update_memory(self, agent_idx, state):
        """Update the agent's memory based on its state and observations."""
        if state[0] == -1:  # Agent is deactivated or evacuated
            return

        # Extract agent position and orientation
        x, y = int(state[0]), int(state[1])
        orientation = int(state[2])

        # Update grid size if necessary based on observations
        self.grid_size = max(self.grid_size, x + 1, y + 1)

        # Make sure memory grid is large enough
        if self.agent_memories[agent_idx].shape[0] < self.grid_size:
            new_memory = np.ones((self.grid_size, self.grid_size)) * -1
            h, w = self.agent_memories[agent_idx].shape
            new_memory[:h, :w] = self.agent_memories[agent_idx]
            self.agent_memories[agent_idx] = new_memory

        # Update agent's current position as free space
        self.agent_memories[agent_idx][x, y] = 0

        # Update goal position
        goal_x, goal_y = int(state[4]), int(state[5])
        self.goal_positions[agent_idx] = (goal_x, goal_y)
        self.agent_memories[agent_idx][goal_x, goal_y] = 4  # Mark goal

        # Process LIDAR observations for main, right, and left
        self.process_lidar_data(agent_idx, x, y, orientation, state[6:12])

        # Process observations from other agents in communication range
        start_idx = 12
        for i in range(self.num_agents - 1):
            idx = start_idx + i * 10
            if idx + 10 <= len(state) and state[idx] != -1:
                # Extract other agent position and update memory
                other_x, other_y = int(state[idx]), int(state[idx + 1])
                other_orientation = int(state[idx + 2])
                self.agent_memories[agent_idx][other_x, other_y] = (
                    3  # Mark as another agent
                )

                # Process the other agent's LIDAR data if available
                self.process_other_agent_lidar(
                    agent_idx,
                    other_x,
                    other_y,
                    other_orientation,
                    state[idx + 4 : idx + 10],
                )


    def update_danger_zones(self, agent_idx):
        """Remove expired danger zones and update memory accordingly."""
        expired_cells = []
        
        # Find expired danger zones
        for cell, [timestamp, obstacle_type] in self.danger_zones[agent_idx].items():
            if self.time_step - timestamp > self.danger_zone_expiry:
                expired_cells.append(cell)
                
                # Update memory to free space for expired danger zones
                # only if they're not currently detected as obstacles
                if (self.agent_memories[agent_idx][cell] == 2 or 
                    self.agent_memories[agent_idx][cell] == -1):
                    self.agent_memories[agent_idx][cell] = 0
        
        # Remove expired cells from danger zones
        for cell in expired_cells:
            del self.danger_zones[agent_idx][cell]

    def process_lidar_data(self, agent_idx, x, y, orientation, lidar_data):
        """Process the LIDAR data and update memory."""
        # Define directions based on orientation
        directions = {
            0: [(-1, 0), (0, 1), (0, -1)],  # Up, Right, Left
            1: [(0, 1), (1, 0), (-1, 0)],  # Right, Down, Up
            2: [(1, 0), (0, -1), (0, 1)],  # Down, Left, Right
            3: [(0, -1), (-1, 0), (1, 0)],  # Left, Up, Down
        }

        # Process each LIDAR reading (main, right, left)
        for i, (dx, dy) in enumerate(directions[orientation]):
            distance = lidar_data[i * 2]  # Distance to obstacle
            obstacle_type = int(lidar_data[i * 2 + 1])  # Type of obstacle

            # Mark all cells along the LIDAR beam as free until an obstacle
            for d in range(1, int(distance)):
                nx, ny = x + dx * d, y + dy * d
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    self.agent_memories[agent_idx][nx, ny] = 0  # Mark as free space

            # Mark the obstacle if one was detected
            if obstacle_type > 0:
                obstacle_x = x + dx * int(distance)
                obstacle_y = y + dy * int(distance)
                if (
                    0 <= obstacle_x < self.grid_size
                    and 0 <= obstacle_y < self.grid_size
                ):
                    self.agent_memories[agent_idx][obstacle_x, obstacle_y] = (
                        obstacle_type
                    )

                    # If it's a dynamic obstacle, mark surrounding cells and path cells as danger zone
                    if obstacle_type == 2:
                        # Mark the surrounding cells (8-connected neighborhood)
                        for nx in range(obstacle_x - 1, obstacle_x + 2):
                            for ny in range(obstacle_y - 1, obstacle_y + 2):
                                if (
                                    0 <= nx < self.grid_size
                                    and 0 <= ny < self.grid_size
                                ):
                                    # Add with timestamp
                                    self.danger_zones[agent_idx][(nx, ny)] = [self.time_step, obstacle_type]
                        
                        # Also mark cells between agent and obstacle as danger zone
                        for d in range(1, int(distance)):
                            corridor_x = x + dx * d
                            corridor_y = y + dy * d
                            if (
                                0 <= corridor_x < self.grid_size
                                and 0 <= corridor_y < self.grid_size
                            ):
                                # Add with timestamp
                                self.danger_zones[agent_idx][(corridor_x, corridor_y)] = [self.time_step, obstacle_type]

    def process_other_agent_lidar(
        self, agent_idx, other_x, other_y, other_orientation, lidar_data
    ):
        """Process LIDAR data from other agents to update memory."""
        # Define directions based on orientation
        directions = {
            0: [(-1, 0), (0, 1), (0, -1)],  # Up, Right, Left
            1: [(0, 1), (1, 0), (-1, 0)],  # Right, Down, Up
            2: [(1, 0), (0, -1), (0, 1)],  # Down, Left, Right
            3: [(0, -1), (-1, 0), (1, 0)],  # Left, Up, Down
        }

        # Process each LIDAR reading (main, right, left)
        for i, (dx, dy) in enumerate(directions[other_orientation]):
            distance = lidar_data[i * 2]  # Distance to obstacle
            obstacle_type = int(lidar_data[i * 2 + 1])  # Type of obstacle

            # Mark all cells along the LIDAR beam as free until an obstacle
            for d in range(1, int(distance)):
                nx, ny = other_x + dx * d, other_y + dy * d
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    # Only update if current value is unknown
                    if self.agent_memories[agent_idx][nx, ny] == -1:
                        self.agent_memories[agent_idx][nx, ny] = 0

            # Mark the obstacle if one was detected
            if obstacle_type > 0:
                obstacle_x = other_x + dx * int(distance)
                obstacle_y = other_y + dy * int(distance)
                if (
                    0 <= obstacle_x < self.grid_size
                    and 0 <= obstacle_y < self.grid_size
                ):
                    # Only update if current value is unknown or free
                    curr_val = self.agent_memories[agent_idx][obstacle_x, obstacle_y]
                    if curr_val == -1 or curr_val == 0:
                        self.agent_memories[agent_idx][obstacle_x, obstacle_y] = (
                            obstacle_type
                        )

                    # If it's a dynamic obstacle, mark surrounding cells as danger zone with timestamps
                    if obstacle_type == 2:
                        for nx in range(obstacle_x - 1, obstacle_x + 2):
                            for ny in range(obstacle_y - 1, obstacle_y + 2):
                                if (
                                    0 <= nx < self.grid_size
                                    and 0 <= ny < self.grid_size
                                ):
                                    # Add with timestamp
                                    self.danger_zones[agent_idx][(nx, ny)] = [self.time_step, obstacle_type]
                        
                        # Also mark cells between other agent and obstacle as danger zone
                        for d in range(1, int(distance)):
                            corridor_x = other_x + dx * d
                            corridor_y = other_y + dy * d
                            if (
                                0 <= corridor_x < self.grid_size
                                and 0 <= corridor_y < self.grid_size
                            ):
                                # Add with timestamp
                                self.danger_zones[agent_idx][(corridor_x, corridor_y)] = [self.time_step, obstacle_type]

            

    def a_star(self, agent_idx, start, goal):
        """A* pathfinding algorithm to find optimal path."""
        if start == goal:
            return [start]

        memory = self.agent_memories[agent_idx]

        # Heuristic function (Manhattan distance)
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = defaultdict(lambda: float("inf"))
        g_score[start] = 0
        f_score = defaultdict(lambda: float("inf"))
        f_score[start] = heuristic(start, goal)

        open_set_hash = {start}

        while open_set:
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            # Check all adjacent cells
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)

                # Check if neighbor is within grid bounds
                if not (
                    0 <= neighbor[0] < self.grid_size
                    and 0 <= neighbor[1] < self.grid_size
                ):
                    continue

                # Check if neighbor is traversable
                if (
                    memory[neighbor] == 1 or memory[neighbor] == 2
                ):  # Wall or dynamic obstacle
                    continue

                # Check if neighbor is in danger zone (adjacent to dynamic obstacle)
                if neighbor in self.danger_zones[agent_idx]:
                    continue

                # Calculate tentative g_score
                tentative_g_score = g_score[current] + 1

                if tentative_g_score < g_score[neighbor]:
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        # No path found, try a less restrictive search that allows for unknown cells
        return self.a_star_allow_unknown(agent_idx, start, goal)

    def a_star_allow_unknown(self, agent_idx, start, goal):
        """A* pathfinding that allows for unknown cells when no path is found."""
        memory = self.agent_memories[agent_idx]

        # Heuristic function (Manhattan distance)
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = defaultdict(lambda: float("inf"))
        g_score[start] = 0
        f_score = defaultdict(lambda: float("inf"))
        f_score[start] = heuristic(start, goal)

        open_set_hash = {start}

        while open_set:
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            # Check all adjacent cells
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)

                # Check if neighbor is within grid bounds
                if not (
                    0 <= neighbor[0] < self.grid_size
                    and 0 <= neighbor[1] < self.grid_size
                ):
                    continue

                # Only avoid known obstacles (walls)
                if memory[neighbor] == 1:
                    continue

                # Add penalty for unknown cells and dynamic obstacles
                penalty = 0
                if memory[neighbor] == -1:  # Unknown
                    penalty = 2
                elif memory[neighbor] == 2:  # Dynamic obstacle
                    penalty = 5
                elif neighbor in self.danger_zones[agent_idx]:
                    penalty = 5

                # Calculate tentative g_score with penalty
                tentative_g_score = g_score[current] + 1 + penalty

                if tentative_g_score < g_score[neighbor]:
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        # No path found, return a path toward the goal even if it's not optimal
        return self.greedy_path(start, goal)

    def greedy_path(self, start, goal):
        """Find a greedy path toward the goal when A* fails."""
        path = [start]
        current = start

        for _ in range(100):  # Limit iterations to avoid infinite loops
            best_dist = float("inf")
            best_next = None

            # Check all adjacent cells
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_pos = (current[0] + dx, current[1] + dy)

                # Check if neighbor is within grid bounds
                if not (
                    0 <= next_pos[0] < self.grid_size
                    and 0 <= next_pos[1] < self.grid_size
                ):
                    continue

                dist = abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])
                if dist < best_dist:
                    best_dist = dist
                    best_next = next_pos

            if best_next is None:
                break

            path.append(best_next)
            current = best_next

            if current == goal:
                break

        return path

    def get_action(self, states_list, evaluation=False):
        """Determine actions for all agents based on their current state and memory."""
        # Increment the time step
        self.time_step += 1
        actions = []

        for agent_idx, state in enumerate(states_list):
            if state[0] == -1 or agent_idx in set(range(len(states_list))) - set(
                range(self.num_agents)
            ):
                # Agent is deactivated, evacuated, or out of range
                actions.append(0)  # Stay steady
                continue

            # Update agent's memory based on its state
            self.update_memory(agent_idx, state)

            # Update danger zones (remove expired ones)
            self.update_danger_zones(agent_idx)

            # Get agent position and orientation
            x, y = int(state[0]), int(state[1])
            orientation = int(state[2])
            current_pos = (x, y)

            # Check if agent is stuck in the same position
            if self.previous_positions[agent_idx] == current_pos:
                self.stuck_counter[agent_idx] += 1
            else:
                self.stuck_counter[agent_idx] = 0

            self.previous_positions[agent_idx] = current_pos

            # If agent is stuck for too long, take a random action
            if self.stuck_counter[agent_idx] > 5:
                # Choose a random action to unstick
                action = np.random.choice([1, 2, 3, 4, 5, 6])
                self.stuck_counter[agent_idx] = 0
                actions.append(action)
                self.last_actions[agent_idx] = action
                continue

            # Check if we've reached the goal
            if current_pos == self.goal_positions[agent_idx]:
                actions.append(0)  # Stay steady if at the goal
                continue

            # Calculate path using A*
            goal = self.goal_positions[agent_idx]
            path = self.a_star(agent_idx, current_pos, goal)
            self.agent_paths[agent_idx] = path

            # Determine action based on path
            if len(path) > 1:
                next_pos = path[1]

                # Determine required direction to move
                dx, dy = next_pos[0] - current_pos[0], next_pos[1] - current_pos[1]

                # Determine required orientation
                target_orientation = None
                if dx == -1:  # Need to go up
                    target_orientation = 0
                elif dx == 1:  # Need to go down
                    target_orientation = 2
                elif dy == -1:  # Need to go left
                    target_orientation = 3
                elif dy == 1:  # Need to go right
                    target_orientation = 1

                # If we need to rotate, do that first
                if orientation != target_orientation:
                    # Determine if we should rotate left or right
                    if (orientation + 1) % 4 == target_orientation:
                        action = 5  # Turn right
                    else:
                        action = 6  # Turn left
                else:
                    # We're already facing the right direction, move forward
                    # Map orientation to appropriate movement action
                    if orientation == 0:  # Facing up
                        action = 1  # Move up
                    elif orientation == 1:  # Facing right
                        action = 4  # Move right
                    elif orientation == 2:  # Facing down
                        action = 2  # Move down
                    elif orientation == 3:  # Facing left
                        action = 3  # Move left
            else:
                # No valid path or already at goal, stay steady
                action = 0

            # Check if we're repeating the same action too much
            if action == self.last_actions[agent_idx]:
                self.action_repeat_counter[agent_idx] += 1
                if self.action_repeat_counter[agent_idx] > self.max_repeat_count:
                    # Try a different action
                    options = [1, 2, 3, 4, 5, 6]
                    if action in options:
                        options.remove(action)
                    action = np.random.choice(options)
                    self.action_repeat_counter[agent_idx] = 0
            else:
                self.action_repeat_counter[agent_idx] = 0

            self.last_actions[agent_idx] = action
            actions.append(action)

        return actions

    def update_policy(self, states, actions, rewards, next_states, done, env=None):
        """Update the agent's policy (not used for rule-based agent)."""
        # This is a rule-based agent, so no learning updates are needed
        pass

    def new_episode(self):
        """Reset the agent for a new episode."""
        # Reset all memories and paths
        self.initialize_memories()
        self.agent_paths = [[] for _ in range(self.num_agents)]
        self.previous_positions = [None] * self.num_agents
        self.stuck_counter = [0] * self.num_agents
        self.last_actions = [0] * self.num_agents
        self.action_repeat_counter = [0] * self.num_agents
        self.danger_zones = [set() for _ in range(self.num_agents)]
