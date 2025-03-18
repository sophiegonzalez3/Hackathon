import gymnasium as gym
import numpy as np
import pygame
import math
from gymnasium import spaces
from collections import deque
from typing import Tuple, List, Set, Optional, Union, Dict
from reward import compute_reward

class MazeEnv(gym.Env):

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 4
    }

    def __init__(self, size=30, walls_proportion=0.6, num_dynamic_obstacles=0, num_agents=4, communication_range: int = 8, max_lidar_dist_main: int = 5, max_lidar_dist_second: int = 3, max_episode_steps: int = 500, render_mode='human', 
                 seed: Optional[int] = None):
        super().__init__()
        
        ### CONFIG ###

        # Grid & graphic
        self.grid_size = size
        self.window_size = 512
        self.cell_size = self.window_size // self.grid_size
        self.render_mode = render_mode
        self.screen_size = self.window_size
        self.corners = [(0, 0), (0, self.grid_size-1), 
                       (self.grid_size-1, 0), (self.grid_size-1, self.grid_size-1)]   # Define grid's corners
        self.walls_proportion = walls_proportion
        self.num_dynamic_obstacles = num_dynamic_obstacles   # Dynamic obstacles
        self.colors = {
            'empty': (215, 215, 215),
            'wall': (40, 40, 40),
            'goal': (120, 244, 120),
            'white': (255, 255, 255),
            'obstacle': (147, 112, 219),
            'agents': [
                (100, 149, 237),
                (205, 92, 92),
                (60, 179, 113),
                (240, 230, 140)
            ],
            'lidar': [
                (100, 149, 237, 128),
                (205, 92, 92, 128),
                (60, 179, 113, 128),
                (240, 230, 140, 128)
            ]
        }
        self.directions = {
            0: (0, 0),   # Steady
            1: (-1, 0),  # Up
            2: (1, 0),   # Down
            3: (0, -1),  # Left
            4: (0, 1)    # Right
        }

        # Agents
        self.num_agents = num_agents
        self.communication_range = communication_range
        self.max_lidar_dist_main = max_lidar_dist_main
        self.max_lidar_dist_second = max_lidar_dist_second
        self.action_space = spaces.Discrete(7)
        self.single_agent_state_size = 3 + 3  + 2 * 3 + 4 * (num_agents - 1) + 6 * (num_agents - 1)  # State space for a single agent
        # agent pos, goal pos, LIDAR obs (3 directions dist, obstacle found), other agent positions + LIDAR obs
        self.lidar_directions = {
            0: [(-1, 0), (0, 1), (0, -1)],   # Up main, Right then Left second
            1: [(0, 1), (1, 0), (-1, 0)],    # Left main, Down then Up second
            2: [(1, 0), (0, -1), (0, 1)],    # Down main, Left then Right second
            3: [(0, -1), (-1, 0), (1, 0)]    # Right main, Up then Down second
        }

        # Initialization of the random number generaton
        self.np_random = None
        self.seed_value = None
        if seed:
            self.seed(seed)
        
        # Setup Pygame
        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
        
        # Initial state
        self.dynamic_obstacles = []
        self.deactivated_agents = set()
        self.evacuated_agents = set()
        self.agent_positions = []
        self.lidar_orientation = np.zeros(self.num_agents)
        self.start_positions = []
        self.goal_area = []
        self.walls = set()
        self.grid = None
        self.max_episode_steps = max_episode_steps
        self.current_step = 0


    def set_start_and_goal(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Set start positions and goal areas for all agents using seeded random."""
        start_corner = self.corners[np.random.randint(0, len(self.corners))]
        opposite_corner = (
            self.grid_size - 1 - start_corner[0],
            self.grid_size - 1 - start_corner[1]
        )
        start_positions = [
            (start_corner[0] - i % 2 if start_corner[0] == self.grid_size - 1 else start_corner[0] + i % 2,
            start_corner[1] - i // 2 if start_corner[1] == self.grid_size - 1 else start_corner[1] + i // 2)
            for i in range(self.num_agents)
        ]
        goal_area = [
            (opposite_corner[0] - i % 2 if opposite_corner[0] == self.grid_size - 1 else opposite_corner[0] + i % 2,
            opposite_corner[1] - i // 2 if opposite_corner[1] == self.grid_size - 1 else opposite_corner[1] + i // 2)
            for i in range(self.num_agents)
        ]
        return start_positions, goal_area


    def initialize_dynamic_obstacles(self):
        """
        Initialize dynamic obstacles with seeded random positions,
        ensuring they are at least 15 cells away from start and goal positions.
        """
        dynamic_obstacles = []
        positions = set()
        max_attempts = 1000  # To avoid infinite loop
        attempts = 0
        
        def manhattan_distance(pos1, pos2):
            """Calculate Manhattan distance between two positions"""
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        def is_position_valid(pos):
            """Check if a position respects all constraints"""
            # Check if pos is already filled
            if (pos in self.walls or 
                pos in positions or 
                pos in self.start_positions or 
                pos in self.goal_area):
                return False
            
            # Check minimal distance with start positions
            for start_pos in self.start_positions:
                if manhattan_distance(pos, start_pos) < 12:
                    return False
            
            # Check minimal distance with goal positions
            for goal_pos in self.goal_area:
                if manhattan_distance(pos, goal_pos) < 10:
                    return False
            
            return True
        
        while len(dynamic_obstacles) < self.num_dynamic_obstacles and attempts < max_attempts:
            x = self.np_random.integers(0, self.grid_size)  # Changed from randint to integers
            y = self.np_random.integers(0, self.grid_size)  # Changed from randint to integers
            pos = (x, y)
            
            if is_position_valid(pos):
                dynamic_obstacles.append(pos)
                positions.add(pos)
            
            attempts += 1
  
        # If we didn't achieve the setup of all obstacles, reduce the number of obstacles
        if len(dynamic_obstacles) < self.num_dynamic_obstacles:
            print(f"Warning: Could only place {len(dynamic_obstacles)} dynamic obstacles "
                f"instead of {self.num_dynamic_obstacles} due to space constraints")
        
        return dynamic_obstacles


    def update_dynamic_obstacles(self):
        """Move dynamic obstacles using seeded random."""
        new_dynamic_obstacles = []
        for obstacle in self.dynamic_obstacles:
            dx, dy = self.directions[self.np_random.integers(0, len(self.directions))]  # Changed from randint to integers
            new_x, new_y = obstacle[0] + dx, obstacle[1] + dy
            if (0 <= new_x < self.grid_size and 
                0 <= new_y < self.grid_size and
                (new_x, new_y) not in self.walls and
                (new_x, new_y) not in self.start_positions and
                (new_x, new_y) not in self.goal_area and
                (new_x, new_y) not in self.dynamic_obstacles):
                new_dynamic_obstacles.append((new_x, new_y))
            else:
                new_dynamic_obstacles.append(obstacle)
        self.dynamic_obstacles = new_dynamic_obstacles


    def lidar_scan(self, agent_idx: int, pos: Tuple[int, int]) -> List[int]:
        """Perform LIDAR scan to return seen obstacles and distance."""
        results = []
        for idx, (dx, dy) in enumerate(self.lidar_directions[self.lidar_orientation[agent_idx]]):   # Check in the directions of the current LIDAR orientation

            if idx == 0:
                max_lidar_dist = self.max_lidar_dist_main
            else:
                max_lidar_dist = self.max_lidar_dist_second

            x, y = pos[0], pos[1]
            found_obstacle = False

            for dist in range(max_lidar_dist):  # Scan up to max_lidar_dist cells
                new_x, new_y = x + dx, y + dy

                if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size) or (self.grid[new_x, new_y] == 1):   # Hit map boundaries or Wall
                    results.extend([new_x, new_y, 1])
                    found_obstacle = True
                    break
                elif (new_x, new_y) in self.dynamic_obstacles:   # Dynamic obstacle
                    results.extend([new_x, new_y, 2])
                    found_obstacle = True
                    break
                elif any(np.array_equal(np.array([new_x, new_y]), agent_pos) for agent_pos in self.agent_positions):   # Another agent
                    results.extend([new_x, new_y, 3])
                    found_obstacle = True
                    break
                x, y = new_x, new_y

            if not found_obstacle:
                results.extend([x, y, 0])   # Nothing found; no obstacle

        return results


    def get_agent_state(self, agent_idx: int) -> np.ndarray:
        """Get state for a specific agent: self position, goal position, lidar scan, and other close agents positions"""
        state = np.full(self.single_agent_state_size, -1, dtype=np.float32)   # Initialization with -1
        
        # If the agent is deactivated or evacuated, return state with -1
        if agent_idx in self.deactivated_agents or agent_idx in self.evacuated_agents:
            return state
        
        # Agent position
        agent_pos = np.array(self.agent_positions[agent_idx])
        state[0:2] = agent_pos
        state[2] = self.lidar_orientation[agent_idx]

        # Agent status (evacuated, deactivated, running)
        if agent_idx in self.evacuated_agents:
            state[3] = 1
        elif agent_idx in self.deactivated_agents:
            state[3] = 2
        else:
            state[3] = 0

        # Goal position
        goal_pos = self.goal_area[agent_idx]
        state[4:6] = goal_pos

        # LIDAR state
        for i in range(3):  # Main direction & 2 others directions
            obstacle_pos = np.array([self.lidar_data[agent_idx][3 * i], self.lidar_data[agent_idx][3 * i + 1]])
            distance_to_obstacle = np.linalg.norm(obstacle_pos - agent_pos)
            state[6 + i * 2] = distance_to_obstacle
            state[6 + i * 2 + 1] = self.lidar_data[agent_idx][3 * i + 2]

        # Other agents positions and LIDAR data within communication range
        added_agents_count = 0

        for i, other_pos in enumerate(self.agent_positions):
            if i!= agent_idx:
                distance = np.linalg.norm(agent_pos - other_pos)
                if distance < self.communication_range:   # Communication only if in communication range
                    base_index = 11 + added_agents_count * 2
                    # Insert other agent's position
                    state[(12 + added_agents_count * 10):(12 + added_agents_count * 10 + 2)] = other_pos
                    state[(12 + added_agents_count * 10 + 2)] = self.lidar_orientation[i]
                    # Insert other agent's status
                    if i in self.evacuated_agents:
                        state[(12 + added_agents_count * 10 + 3)] = 1
                    elif i in self.deactivated_agents:
                        state[(12 + added_agents_count * 10 + 3)] = 2
                    else:
                        state[(12 + added_agents_count * 10 + 3)] = 0
                    # Insert LIDAR data of the other agent
                    for j in range(3):
                        other_obstacle_pos = np.array([self.lidar_data[i][3 * j], self.lidar_data[i][3 * j + 1]])
                        distance_to_other_obstacle = np.linalg.norm(other_obstacle_pos - other_pos)
                        state[(12 + added_agents_count * 10 + 4 + j * 2)] = distance_to_other_obstacle
                        state[(12 + added_agents_count * 10 + 4 + j * 2 + 1)] = self.lidar_data[i][3 * j + 2]
                    added_agents_count += 1

        return state

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Environment reset with a new optional seed."""
        self.current_step = 0
        if seed is not None:
            self.seed(seed)
        
        rng_state = np.random.get_state()
        np.random.seed(self.seed_value)
        
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.start_positions, self.goal_area = self.set_start_and_goal()
        self.walls = self.generate_city_layout_with_solution()
        
        for wall in self.walls:
            self.grid[wall] = 1
        
        self.dynamic_obstacles = self.initialize_dynamic_obstacles()
        for obstacle in self.dynamic_obstacles:
            self.grid[obstacle] = 3
        
        self.evacuated_agents = set()
        self.deactivated_agents = set()
        
        self.agent_positions = []
        for i, pos in enumerate(self.start_positions):
            self.grid[pos] = i + 2
            self.agent_positions.append(np.array(pos))

        self.lidar_orientation = np.full(self.num_agents, 
                                         0 if (self.start_positions[0][0] > self.goal_area[0][0]) else 2)   # Check if goal under or above
        self.lidar_data = [self.lidar_scan(agent_idx, tuple(self.agent_positions[agent_idx])) for agent_idx in range(self.num_agents)]

        for pos in self.goal_area:
            if self.grid[pos] == 0:
                self.grid[pos] = -1
        
        np.random.set_state(rng_state)
        
        # Create the initial state for each agent
        state = np.array([self.get_agent_state(i) for i in range(self.num_agents)])
        
        info = {'seed': self.seed_value}
        
        if self.render_mode == "human":
            self._render_frame()
        
        return state, info


    def get_reward(self, old_positions: list):
        rewards, evacuated_agents = compute_reward(self.num_agents, old_positions,
                                                   self.agent_positions, self.evacuated_agents, 
                                                   self.deactivated_agents, self.goal_area)
        if evacuated_agents != self.evacuated_agents:
            self.evacuated_agents = evacuated_agents    

        return rewards


    def step(self, actions):
        self.current_step += 1
        
        # Store the actual state for reward computation
        old_positions = [pos.copy() for pos in self.agent_positions]
        
        # Update dynamic obstacles
        self.update_dynamic_obstacles()

        # Process actions
        if not isinstance(actions, (list, tuple)):
            actions = [actions]

        def check_move_valid(agent_idx, old_position, new_position):
            """
            Vérifie si un agent s'est déplacé de plus d'une unité dans l'une des quatre directions.
            """
            x_old, y_old = old_position
            x_new, y_new = new_position

            # Calcul de la différence dans les directions X et Y
            delta_x = abs(x_new - x_old)
            delta_y = abs(y_new - y_old)

            # Vérification si le mouvement dépasse 1 unité dans l'une des directions
            if delta_x > 1 or delta_y > 1:
                return False  # Mouvement invalide, plus de 1 unité dans l'une des directions
            return True

        # Compute proposed positions
        proposed_positions = []
        for i, (action, agent_pos) in enumerate(zip(actions, self.agent_positions)):
            if 0 <= action <= 4:
                if i in self.deactivated_agents or i in self.evacuated_agents:
                    new_pos = agent_pos   # Steady
                    proposed_positions.append(new_pos)
                    continue
                new_pos = agent_pos + self.directions[action]   # New position
                
            if action == 5:   # Rotate right
                self.lidar_orientation[i] = (self.lidar_orientation[i] + 1) % 4
                new_pos = agent_pos   # Steady
            if action == 6: # Rotate left
                self.lidar_orientation[i] = (self.lidar_orientation[i] - 1) % 4
                new_pos = agent_pos   # Steady

            proposed_positions.append(new_pos)
        
            if not check_move_valid(i, agent_pos, new_pos):
                print('WARNING')
                print(i, agent_pos, proposed_positions)
        # Resolve collisions
        self.resolve_collisions(proposed_positions)
        
        # Get LIDAR scan
        self.lidar_data = [self.lidar_scan(agent_idx, tuple(self.agent_positions[agent_idx])) for agent_idx in range(self.num_agents)]

        # Update the grid to display new positions
        self.update_grid()
        
        episode_ending = False
        truncated = False
        
        # Check conditions for the end of the episode
        if self.current_step >= self.max_episode_steps:
            episode_ending = True
            truncated = True
        
        # Check if all agents are evacuated or deactivated
        if len(self.deactivated_agents | self.evacuated_agents) == self.num_agents:
            episode_ending = True

        rewards = self.get_reward(old_positions)
        
        # Prepare state and informations
        state = np.array([self.get_agent_state(i) for i in range(self.num_agents)])
        
        info = {
            'individual_rewards': rewards,
            'deactivated_agents': list(self.deactivated_agents),
            'evacuated_agents': list(self.evacuated_agents),
            'current_step': self.current_step,
            'max_episode_steps': self.max_episode_steps
        }

        if self.render_mode == "human":
            self._render_frame()
        
        return state, rewards, episode_ending, truncated, info


    def update_grid(self):
        """Update grid with actual positions"""
        # Reinitialize grid with static elements
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for wall in self.walls:
            self.grid[wall] = 1
        for pos in self.goal_area:
            if self.grid[pos] == 0:  # Avoid overwriting the walls
                self.grid[pos] = -1
                
        # Reinitialize dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            self.grid[obstacle] = 3
            
        # Update agent positions
        for i, pos in enumerate(self.agent_positions):
            if i not in self.deactivated_agents and i not in self.evacuated_agents:
                self.grid[tuple(pos)] = i + 2


    def resolve_collisions(self, proposed_positions: List[np.ndarray]) -> None:
        # Compute priorities only for active agents
        priorities = []
        
        for i, pos in enumerate(proposed_positions):
            if i in self.evacuated_agents or i in self.deactivated_agents:
                priorities.append(float('inf'))
                continue
            min_dist_to_goal = min(np.linalg.norm(pos - np.array(goal)) for goal in self.goal_area)
            priorities.append(min_dist_to_goal)

        agent_order = sorted(range(len(priorities)), key=lambda k: priorities[k])

        new_positions = [pos.copy() for pos in self.agent_positions]

        def is_valid_position(pos, agent_idx):
            """ Return the type of collision for a given position"""
            if not (0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size):   # Check boundaries
                return False, "out_of_bounds"

            pos_tuple = tuple(pos.astype(int))

            # Check walls
            if pos_tuple in self.walls:
                return False, "wall"

            # Check dynamic obstacles
            if pos_tuple in self.dynamic_obstacles:
                return False, "dynamic_obstacle"
            x, y = pos
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                adjacent_pos = (x + dx, y + dy)
                if adjacent_pos in self.dynamic_obstacles:
                    return False, "dynamic_obstacle"

            # Check other agents
            for i, other_pos in enumerate(new_positions):
                if i != agent_idx and i not in self.evacuated_agents and i not in self.deactivated_agents:
                    if np.array_equal(pos.astype(int), other_pos.astype(int)):
                        return False, "agent"

            return True, None

        # Resolve collisions by priority order
        for idx in agent_order:
            if idx in self.evacuated_agents or idx in self.deactivated_agents:
                continue
            
            valid, col_type = is_valid_position(proposed_positions[idx], idx)
            # Si la position proposée est valide, l'utiliser
            if valid:
                new_positions[idx] = proposed_positions[idx]
                continue
            elif col_type in ["out_of_bounds", "agent"]:
                new_positions[idx] = self.agent_positions[idx]
            elif col_type in ["dynamic_obstacle", "wall"]:
                new_positions[idx] = np.array([-1, -1])
                self.deactivated_agents.add(idx)

        self.agent_positions = new_positions


    def seed(self, seed=None):  
        """Initialize random numbers generator."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.seed_value = seed
        np.random.seed(seed)
        return [seed]


    def generate_city_layout_with_solution(self) -> Set[Tuple[int, int]]:
        """Generate a city layout with guaranteed path to goal using seeded random."""
        max_attempts = 500
        attempt = 0
        
        while attempt < max_attempts:
            walls = set()
            
            for i in range(0, self.grid_size, 4):
                for j in range(self.grid_size):
                    if (self.np_random.random() < self.walls_proportion and 
                        (i, j) not in self.start_positions and 
                        (i, j) not in self.goal_area):
                        walls.add((i, j))
            
            for j in range(0, self.grid_size, 4):
                for i in range(self.grid_size):
                    if (self.np_random.random() < self.walls_proportion and 
                        (i, j) not in self.start_positions and 
                        (i, j) not in self.goal_area):
                        walls.add((i, j))
                    
            if self.is_path_available(walls):
                return walls
                
            attempt += 1
        
        return set()


    def is_path_available(self, walls: Set[Tuple[int, int]]) -> bool:
        """Check if there's a valid path from start to goal."""
        for start_pos in self.start_positions:
            queue = deque([start_pos])
            visited = set()
            found_path = False
            
            while queue and not found_path:
                current = queue.popleft()
                if current in visited:
                    continue
                    
                visited.add(current)
                
                if current in self.goal_area:
                    found_path = True
                    break
                
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    next_x, next_y = current[0] + dx, current[1] + dy
                    neighbor = (next_x, next_y)
                    
                    if (0 <= next_x < self.grid_size and 
                        0 <= next_y < self.grid_size and 
                        neighbor not in walls and 
                        neighbor not in visited):
                            queue.append(neighbor)
            
            if not found_path:
                return False
        
        return True


    def _render_frame(self):
        try:
            self.window.fill(self.colors['empty'])
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return

            # Draw grid and static elements first
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    cell_value = self.grid[i, j]
                    color = self.colors['empty']
                    if cell_value == 1:
                        color = self.colors['wall']
                    elif cell_value == -1:
                        color = self.colors['goal']
                    
                    pygame.draw.rect(
                        self.window,
                        color,
                        pygame.Rect(
                            j * self.cell_size,
                            i * self.cell_size,
                            self.cell_size,
                            self.cell_size
                        )
                    )
                    pygame.draw.rect(
                        self.window,
                        (240, 240, 240),
                        pygame.Rect(
                            j * self.cell_size,
                            i * self.cell_size,
                            self.cell_size,
                            self.cell_size
                        ),
                        1
                    )

            # Create a surface for LIDAR visualization
            lidar_surface = pygame.Surface((self.screen_size, self.screen_size), pygame.SRCALPHA)

            # Draw agents with quadcopter representation and LIDAR rays
            for i, pos in enumerate(self.agent_positions):
                if i in self.evacuated_agents or i in self.deactivated_agents:
                    continue
                    
                center_x = int(pos[1] * self.cell_size + self.cell_size // 2)
                center_y = int(pos[0] * self.cell_size + self.cell_size // 2)

                # LIDAR visualization
                for j, (dx, dy) in enumerate(self.lidar_directions[self.lidar_orientation[i]]):
                    if j == 0:
                        max_lidar_dist = self.max_lidar_dist_main
                    else:
                        max_lidar_dist = self.max_lidar_dist_second

                    if self.lidar_data[i][3 * j + 2] == 0:
                        # No wall detected within 5 cells
                        end_x = center_x + int(dy * max_lidar_dist * self.cell_size)
                        end_y = center_y + int(dx * max_lidar_dist * self.cell_size)
                    else:
                        # Wall detected within the range
                        end_x = int(self.lidar_data[i][3 * j + 1] * self.cell_size + self.cell_size // 2)
                        end_y = int(self.lidar_data[i][3 * j] * self.cell_size + self.cell_size // 2)

                    # Draw semi-transparent LIDAR rays
                    pygame.draw.line(lidar_surface, 
                                self.colors['lidar'][i % len(self.colors['lidar'])],
                                (center_x, center_y),
                                (end_x, end_y),
                                3)

                # Quadcopter representation
                body_size = self.cell_size // 2
                arm_length = self.cell_size // 3
                rotor_radius = self.cell_size // 8
                agent_color = self.colors['agents'][i % len(self.colors['agents'])]

                # Central body (octagon)
                points = []
                for angle in range(0, 360, 45):
                    rad = math.radians(angle)
                    x = center_x + int(math.cos(rad) * body_size // 2)
                    y = center_y + int(math.sin(rad) * body_size // 2)
                    points.append((x, y))
                pygame.draw.polygon(self.window, agent_color, points)

                angles = [45, 135, 225, 315]
                for angle in angles:
                    rad = math.radians(angle)
                    end_x = center_x + int(math.cos(rad) * arm_length)
                    end_y = center_y + int(math.sin(rad) * arm_length)
                    pygame.draw.line(self.window, agent_color,
                                (center_x, center_y),
                                (end_x, end_y),
                                3)

                    # Rotors
                    pygame.draw.circle(self.window, agent_color,
                                    (end_x, end_y),
                                    rotor_radius)
                    pygame.draw.circle(self.window, self.colors['white'],
                                    (end_x, end_y),
                                    rotor_radius // 2)

            # Blit the LIDAR surface onto the main screen
            self.window.blit(lidar_surface, (0, 0))
            
            # Draw dynamic obstacles last to ensure proper layering
            for obstacle in self.dynamic_obstacles:
                center_x = int(obstacle[1] * self.cell_size + self.cell_size // 2)
                center_y = int(obstacle[0] * self.cell_size + self.cell_size // 2)
                
                # Draw rotating triangle for dynamic obstacle
                size = self.cell_size // 2
                points = []
                rotation = pygame.time.get_ticks() / 1000.0
                for i in range(3):
                    angle = rotation + i * 120
                    rad = math.radians(angle)
                    x = center_x + int(math.cos(rad) * size)
                    y = center_y + int(math.sin(rad) * size)
                    points.append((x, y))
                
                pygame.draw.polygon(self.window, self.colors['obstacle'], points)
                
                # Add a small dot in the center
                pygame.draw.circle(self.window, self.colors['white'],
                                (center_x, center_y),
                                self.cell_size // 8)
            
            if self.render_mode == "human":
                pygame.display.flip()
                self.clock.tick(self.metadata["render_fps"])

        except:
            print("\nPygame window closed")
            self.close()
            raise KeyboardInterrupt

    def close(self):
        if self.render_mode == "human":
            pygame.display.quit()
            pygame.quit()