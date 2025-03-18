import numpy as np

class MyAgent:
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.rng = np.random.default_rng()
        
        # Agent tracking
        self.previous_positions = [None] * num_agents
        self.stuck_counters = [0] * num_agents
        self.wait_timers = [0] * num_agents
        self.rotation_timers = [0] * num_agents
        self.current_orientation = [None] * num_agents
        self.direction_counters = [0] * num_agents
        self.movement_phase = [0] * num_agents  # 0: first edge, 1: corner, 2: second edge, 3: approach goal
        
        # Action constants
        self.STAY = 0
        self.MOVE_UP = 1
        self.MOVE_DOWN = 2
        self.MOVE_LEFT = 3
        self.MOVE_RIGHT = 4
        self.ROTATE_RIGHT = 5
        self.ROTATE_LEFT = 6
        
        # Movement constants
        self.directions = {
            0: (0, 0),   # Steady
            1: (-1, 0),  # Up
            2: (1, 0),   # Down
            3: (0, -1),  # Left
            4: (0, 1)    # Right
        }
        
        # Safety thresholds
        self.WALL_SAFETY = 2.0
        self.OBSTACLE_SAFETY = 4.0
        self.AGENT_SAFETY = 2.0
        
        # Grid size estimation
        self.grid_size = 30  # Will be refined based on observations
        
        # Counter for step debugging
        self.step_count = 0
    
    def get_action(self, state: list, evaluation: bool = False):
        """Choose actions for all agents based on current state"""
        actions = np.zeros(self.num_agents, dtype=int)
        
        # Update grid size estimation
        self._update_grid_size(state)
        
        for agent_idx in range(self.num_agents):
            # Skip if agent is evacuated or deactivated
            if state[agent_idx][3] != 0:  # Not running
                actions[agent_idx] = self.STAY
                continue
            
            # Extract state information
            agent_pos = state[agent_idx][0:2]
            orientation = int(state[agent_idx][2])
            goal_pos = state[agent_idx][4:6]
            
            # Store current orientation
            self.current_orientation[agent_idx] = orientation
            
            # Extract LIDAR data
            lidar_data = self._parse_lidar(state[agent_idx])
            
            # Check for immediate danger
            has_danger = self._check_for_danger(lidar_data)
            
            # Update movement phase if needed
            if self.movement_phase[agent_idx] == 0 and self._reached_first_edge(agent_idx, agent_pos, goal_pos):
                self.movement_phase[agent_idx] = 1  # Transition to corner
            elif self.movement_phase[agent_idx] == 1 and self._reached_corner(agent_idx, agent_pos, goal_pos):
                self.movement_phase[agent_idx] = 2  # Transition to second edge
            elif self.movement_phase[agent_idx] == 2 and self._close_to_goal(agent_pos, goal_pos):
                self.movement_phase[agent_idx] = 3  # Transition to goal approach
            
            # Decision logic
            if self.wait_timers[agent_idx] > 0:
                # Currently waiting - priority 1
                actions[agent_idx] = self.STAY
                self.wait_timers[agent_idx] -= 1
            elif has_danger:
                # Detected danger - priority 2
                actions[agent_idx] = self._get_danger_avoidance_action(agent_idx, agent_pos, orientation, lidar_data)
                # Reset direction counter when danger is detected
                self.direction_counters[agent_idx] = 0
            elif self.rotation_timers[agent_idx] > 0:
                # Currently rotating - priority 3
                actions[agent_idx] = self.ROTATE_RIGHT
                self.rotation_timers[agent_idx] -= 1
            elif self._is_agent_stuck(agent_idx, agent_pos):
                # Agent is stuck - priority 4
                actions[agent_idx] = self._get_unstuck_action(agent_idx, agent_pos, orientation, lidar_data)
            else:
                # Normal movement based on phase - priority 5
                if self.movement_phase[agent_idx] == 0:
                    # Moving to first edge
                    actions[agent_idx] = self._get_to_first_edge_action(agent_idx, agent_pos, goal_pos, orientation, lidar_data)
                elif self.movement_phase[agent_idx] == 1:
                    # Moving to corner
                    actions[agent_idx] = self._get_to_corner_action(agent_idx, agent_pos, goal_pos, orientation, lidar_data)
                elif self.movement_phase[agent_idx] == 2:
                    # Moving along second edge
                    actions[agent_idx] = self._get_to_second_edge_action(agent_idx, agent_pos, goal_pos, orientation, lidar_data)
                else:
                    # Approaching goal
                    actions[agent_idx] = self._get_to_goal_action(agent_idx, agent_pos, goal_pos, orientation, lidar_data)
            
            # Store current position for stuck detection
            self.previous_positions[agent_idx] = agent_pos.copy()
        
        self.step_count += 1
        return actions.tolist()
    
    def _update_grid_size(self, state):
        """Update grid size estimation based on agent and goal positions"""
        if len(state) > 0:
            max_coord = 0
            for agent_state in state:
                if agent_state[3] == 0:  # Only active agents
                    agent_pos = agent_state[0:2]
                    goal_pos = agent_state[4:6]
                    max_coord = max(max_coord, 
                                   agent_pos[0], agent_pos[1],
                                   goal_pos[0], goal_pos[1])
            
            if max_coord > 0:
                self.grid_size = int(max_coord * 1.1) + 5  # With margin
    
    def _parse_lidar(self, agent_state):
        """Extract LIDAR data from agent state"""
        lidar = []
        for i in range(3):  # 3 LIDAR directions
            distance = agent_state[6 + i*2]
            obstacle_type = agent_state[7 + i*2]
            lidar.append((distance, obstacle_type))
        return lidar
    
    def _check_for_danger(self, lidar_data):
        """Check if any LIDAR beam detects danger"""
        for dist, obs_type in lidar_data:
            if (obs_type == 1 and dist < self.WALL_SAFETY) or \
               (obs_type == 2 and dist < self.OBSTACLE_SAFETY) or \
               (obs_type == 3 and dist < self.AGENT_SAFETY):
                return True
        return False
    
    def _is_agent_stuck(self, agent_idx, current_pos):
        """Check if agent is stuck in the same position"""
        if self.previous_positions[agent_idx] is None:
            return False
        
        if np.array_equal(current_pos, self.previous_positions[agent_idx]):
            self.stuck_counters[agent_idx] += 1
            return self.stuck_counters[agent_idx] >= 5
        else:
            self.stuck_counters[agent_idx] = 0
            return False
    
    def _get_danger_avoidance_action(self, agent_idx, agent_pos, orientation, lidar_data):
        """Get action to avoid detected danger"""
        # Extract LIDAR data
        main_dist, main_type = lidar_data[0]
        right_dist, right_type = lidar_data[1]
        left_dist, left_type = lidar_data[2]
        
        # Check for dynamic obstacles - just STAY rather than move
        for dist, obs_type in lidar_data:
            if obs_type == 2 and dist < 3.0:  # Dynamic obstacle very close
                self.wait_timers[agent_idx] = 3  # Wait for 3 steps
                return self.STAY
        
        # Calculate safety for each direction
        main_safe = self._is_direction_safe(main_dist, main_type)
        right_safe = self._is_direction_safe(right_dist, right_type)
        left_safe = self._is_direction_safe(left_dist, left_type)
        
        # Choose safest direction
        if main_safe and not (right_safe or left_safe):
            # Only main direction is safe
            return self._orientation_to_action(orientation)
        elif right_safe and not (main_safe or left_safe):
            # Only right is safe
            return self.ROTATE_RIGHT
        elif left_safe and not (main_safe or right_safe):
            # Only left is safe
            return self.ROTATE_LEFT
        elif not (main_safe or right_safe or left_safe):
            # No direction is fully safe, STAY put
            self.wait_timers[agent_idx] = 2
            return self.STAY
        else:
            # Multiple directions safe, choose based on safety score
            scores = [
                (self._safety_score(main_dist, main_type), self._orientation_to_action(orientation)),
                (self._safety_score(right_dist, right_type), self.ROTATE_RIGHT),
                (self._safety_score(left_dist, left_type), self.ROTATE_LEFT)
            ]
            return max(scores, key=lambda x: x[0])[1]
    
    def _is_direction_safe(self, distance, obstacle_type):
        """Check if a direction is safe based on distance and obstacle type"""
        if obstacle_type == 0:  # No obstacle
            return True
        elif obstacle_type == 1:  # Wall
            return distance >= self.WALL_SAFETY
        elif obstacle_type == 2:  # Dynamic obstacle
            return distance >= self.OBSTACLE_SAFETY
        elif obstacle_type == 3:  # Other agent
            return distance >= self.AGENT_SAFETY
        return False
    
    def _safety_score(self, distance, obstacle_type):
        """Calculate safety score for a direction (higher is safer)"""
        if obstacle_type == 0:  # No obstacle
            return 100.0
        elif obstacle_type == 1:  # Wall
            return distance
        elif obstacle_type == 2:  # Dynamic obstacle
            return distance * 0.5  # Lower safety for dynamic obstacles
        elif obstacle_type == 3:  # Other agent
            return distance * 0.8
        return 0
    
    def _get_unstuck_action(self, agent_idx, agent_pos, orientation, lidar_data):
        """Get action to help agent get unstuck"""
        # Reset counters
        self.stuck_counters[agent_idx] = 0
        
        # Extract LIDAR data
        main_dist, main_type = lidar_data[0]
        right_dist, right_type = lidar_data[1]
        left_dist, left_type = lidar_data[2]
        
        # If any direction is very dangerous, just wait
        if (main_type == 2 and main_dist < 2.0) or \
           (right_type == 2 and right_dist < 2.0) or \
           (left_type == 2 and left_dist < 2.0):
            self.wait_timers[agent_idx] = 3
            return self.STAY
        
        # If forward is blocked, try rotating
        if main_type > 0 and main_dist < self.WALL_SAFETY:
            self.rotation_timers[agent_idx] = 2
            return self.ROTATE_RIGHT
        
        # Try moving forward if it seems safe
        if main_type == 0 or main_dist > self.WALL_SAFETY:
            return self._orientation_to_action(orientation)
        
        # Fall back to random rotation
        return self.ROTATE_RIGHT if self.rng.random() < 0.5 else self.ROTATE_LEFT
    
    def _reached_first_edge(self, agent_idx, agent_pos, goal_pos):
        """Check if agent has reached the first edge based on position"""
        # Determine which quadrant the agent starts in
        is_top = agent_pos[0] < self.grid_size / 3
        is_bottom = agent_pos[0] > 2 * self.grid_size / 3
        is_left = agent_pos[1] < self.grid_size / 3
        is_right = agent_pos[1] > 2 * self.grid_size / 3
        
        goal_is_top = goal_pos[0] < self.grid_size / 3
        goal_is_bottom = goal_pos[0] > 2 * self.grid_size / 3
        goal_is_left = goal_pos[1] < self.grid_size / 3
        goal_is_right = goal_pos[1] > 2 * self.grid_size / 3
        
        # For top-left to bottom-right path
        if is_top and is_left and goal_is_bottom and goal_is_right:
            # Check if we've reached the top edge (moving right)
            return agent_pos[0] < 5 and agent_pos[1] > self.grid_size / 3
        
        # For top-right to bottom-left path
        elif is_top and is_right and goal_is_bottom and goal_is_left:
            # Check if we've reached the right edge (moving down)
            return agent_pos[1] > self.grid_size - 5 and agent_pos[0] > self.grid_size / 3
        
        # For bottom-left to top-right path
        elif is_bottom and is_left and goal_is_top and goal_is_right:
            # Check if we've reached the left edge (moving up)
            return agent_pos[1] < 5 and agent_pos[0] < 2 * self.grid_size / 3
        
        # For bottom-right to top-left path
        elif is_bottom and is_right and goal_is_top and goal_is_left:
            # Check if we've reached the bottom edge (moving left)
            return agent_pos[0] > self.grid_size - 5 and agent_pos[1] < 2 * self.grid_size / 3
        
        return False
    
    def _reached_corner(self, agent_idx, agent_pos, goal_pos):
        """Check if agent has reached the corner between edges"""
        # Determine which quadrant the agent starts in
        is_top = agent_pos[0] < self.grid_size / 3
        is_bottom = agent_pos[0] > 2 * self.grid_size / 3
        is_left = agent_pos[1] < self.grid_size / 3
        is_right = agent_pos[1] > 2 * self.grid_size / 3
        
        goal_is_top = goal_pos[0] < self.grid_size / 3
        goal_is_bottom = goal_pos[0] > 2 * self.grid_size / 3
        goal_is_left = goal_pos[1] < self.grid_size / 3
        goal_is_right = goal_pos[1] > 2 * self.grid_size / 3
        
        # For top-left to bottom-right path
        if is_top and is_left and goal_is_bottom and goal_is_right:
            # Check if we've reached the top-right corner
            return agent_pos[0] < 5 and agent_pos[1] > self.grid_size - 5
        
        # For top-right to bottom-left path
        elif is_top and is_right and goal_is_bottom and goal_is_left:
            # Check if we've reached the bottom-right corner
            return agent_pos[0] > self.grid_size - 5 and agent_pos[1] > self.grid_size - 5
        
        # For bottom-left to top-right path
        elif is_bottom and is_left and goal_is_top and goal_is_right:
            # Check if we've reached the top-left corner
            return agent_pos[0] < 5 and agent_pos[1] < 5
        
        # For bottom-right to top-left path
        elif is_bottom and is_right and goal_is_top and goal_is_left:
            # Check if we've reached the bottom-left corner
            return agent_pos[0] > self.grid_size - 5 and agent_pos[1] < 5
        
        return False
    
    def _close_to_goal(self, agent_pos, goal_pos):
        """Check if agent is close enough to directly approach goal"""
        distance = np.linalg.norm(agent_pos - goal_pos)
        return distance < 5.0
    
    def _get_to_first_edge_action(self, agent_idx, agent_pos, goal_pos, orientation, lidar_data):
        """Get action to move to the first edge"""
        # Determine which quadrant the agent starts in
        is_top = agent_pos[0] < self.grid_size / 3
        is_bottom = agent_pos[0] > 2 * self.grid_size / 3
        is_left = agent_pos[1] < self.grid_size / 3
        is_right = agent_pos[1] > 2 * self.grid_size / 3
        
        goal_is_top = goal_pos[0] < self.grid_size / 3
        goal_is_bottom = goal_pos[0] > 2 * self.grid_size / 3
        goal_is_left = goal_pos[1] < self.grid_size / 3
        goal_is_right = goal_pos[1] > 2 * self.grid_size / 3
        
        # Default directions and safety checks
        target_orientation = orientation  # Default to current
        main_dist, main_type = lidar_data[0]
        
        # For top-left to bottom-right path
        if is_top and is_left and goal_is_bottom and goal_is_right:
            # First move up to top edge
            if agent_pos[0] > 3:
                target_orientation = 0  # Face up
            else:
                target_orientation = 3  # Face right to move along top edge
        
        # For top-right to bottom-left path
        elif is_top and is_right and goal_is_bottom and goal_is_left:
            # First move right to right edge
            if agent_pos[1] < self.grid_size - 3:
                target_orientation = 3  # Face right
            else:
                target_orientation = 2  # Face down to move along right edge
        
        # For bottom-left to top-right path
        elif is_bottom and is_left and goal_is_top and goal_is_right:
            # First move left to left edge
            if agent_pos[1] > 3:
                target_orientation = 1  # Face left
            else:
                target_orientation = 0  # Face up to move along left edge
        
        # For bottom-right to top-left path
        elif is_bottom and is_right and goal_is_top and goal_is_left:
            # First move down to bottom edge
            if agent_pos[0] < self.grid_size - 3:
                target_orientation = 2  # Face down
            else:
                target_orientation = 1  # Face left to move along bottom edge
        
        # Rotate if needed
        if orientation != target_orientation:
            return self._get_rotation_action(orientation, target_orientation)
        
        # Check if path is clear before moving
        if main_type == 0 or main_dist > self.WALL_SAFETY:
            self.direction_counters[agent_idx] += 1
            return self._orientation_to_action(orientation)
        else:
            # Path blocked, rotate to find clear path
            return self.ROTATE_RIGHT
    
    def _get_to_corner_action(self, agent_idx, agent_pos, goal_pos, orientation, lidar_data):
        """Get action to move to the corner"""
        # Determine which quadrant the agent starts in
        is_top = agent_pos[0] < self.grid_size / 3
        is_bottom = agent_pos[0] > 2 * self.grid_size / 3
        is_left = agent_pos[1] < self.grid_size / 3
        is_right = agent_pos[1] > 2 * self.grid_size / 3
        
        goal_is_top = goal_pos[0] < self.grid_size / 3
        goal_is_bottom = goal_pos[0] > 2 * self.grid_size / 3
        goal_is_left = goal_pos[1] < self.grid_size / 3
        goal_is_right = goal_pos[1] > 2 * self.grid_size / 3
        
        # Default directions and safety checks
        target_orientation = orientation  # Default to current
        main_dist, main_type = lidar_data[0]
        
        # For top-left to bottom-right path (moving along top edge to top-right corner)
        if is_top and is_left and goal_is_bottom and goal_is_right:
            target_orientation = 3  # Face right
        
        # For top-right to bottom-left path (moving along right edge to bottom-right corner)
        elif is_top and is_right and goal_is_bottom and goal_is_left:
            target_orientation = 2  # Face down
        
        # For bottom-left to top-right path (moving along left edge to top-left corner)
        elif is_bottom and is_left and goal_is_top and goal_is_right:
            target_orientation = 0  # Face up
        
        # For bottom-right to top-left path (moving along bottom edge to bottom-left corner)
        elif is_bottom and is_right and goal_is_top and goal_is_left:
            target_orientation = 1  # Face left
        
        # Rotate if needed
        if orientation != target_orientation:
            return self._get_rotation_action(orientation, target_orientation)
        
        # Check if path is clear before moving
        if main_type == 0 or main_dist > self.WALL_SAFETY:
            self.direction_counters[agent_idx] += 1
            return self._orientation_to_action(orientation)
        else:
            # Path blocked, rotate to find clear path
            return self.ROTATE_RIGHT
    
    def _get_to_second_edge_action(self, agent_idx, agent_pos, goal_pos, orientation, lidar_data):
        """Get action to move along the second edge toward goal quadrant"""
        # Determine which quadrant the agent starts in
        is_top = agent_pos[0] < self.grid_size / 3
        is_bottom = agent_pos[0] > 2 * self.grid_size / 3
        is_left = agent_pos[1] < self.grid_size / 3
        is_right = agent_pos[1] > 2 * self.grid_size / 3
        
        goal_is_top = goal_pos[0] < self.grid_size / 3
        goal_is_bottom = goal_pos[0] > 2 * self.grid_size / 3
        goal_is_left = goal_pos[1] < self.grid_size / 3
        goal_is_right = goal_pos[1] > 2 * self.grid_size / 3
        
        # Default directions and safety checks
        target_orientation = orientation  # Default to current
        main_dist, main_type = lidar_data[0]
        
        # For top-left to bottom-right path (from top-right corner to bottom-right)
        if is_top and is_right and goal_is_bottom and goal_is_right:
            target_orientation = 2  # Face down
        
        # For top-right to bottom-left path (from bottom-right corner to bottom-left)
        elif is_bottom and is_right and goal_is_bottom and goal_is_left:
            target_orientation = 1  # Face left
        
        # For bottom-left to top-right path (from top-left corner to top-right)
        elif is_top and is_left and goal_is_top and goal_is_right:
            target_orientation = 3  # Face right
        
        # For bottom-right to top-left path (from bottom-left corner to top-left)
        elif is_bottom and is_left and goal_is_top and goal_is_left:
            target_orientation = 0  # Face up
        
        # Rotate if needed
        if orientation != target_orientation:
            return self._get_rotation_action(orientation, target_orientation)
        
        # Check if path is clear before moving
        if main_type == 0 or main_dist > self.WALL_SAFETY:
            self.direction_counters[agent_idx] += 1
            return self._orientation_to_action(orientation)
        else:
            # Path blocked, rotate to find clear path
            return self.ROTATE_RIGHT
    
    def _get_to_goal_action(self, agent_idx, agent_pos, goal_pos, orientation, lidar_data):
        """Get action to approach goal directly"""
        # Calculate direction to goal
        dx = goal_pos[0] - agent_pos[0]
        dy = goal_pos[1] - agent_pos[1]
        
        # Determine optimal orientation
        if abs(dx) > abs(dy):
            # Prioritize x-axis movement
            target_orientation = 2 if dx > 0 else 0  # Down or Up
        else:
            # Prioritize y-axis movement
            target_orientation = 3 if dy > 0 else 1  # Right or Left
        
        # Extract LIDAR data
        main_dist, main_type = lidar_data[0]
        
        # Rotate if needed
        if orientation != target_orientation:
            return self._get_rotation_action(orientation, target_orientation)
        
        # Check if path is clear before moving
        if main_type == 0 or main_dist > self.WALL_SAFETY:
            return self._orientation_to_action(orientation)
        else:
            # Path blocked, rotate to find clear path
            return self.ROTATE_RIGHT
    
    def _get_rotation_action(self, current, target):
        """Get action to rotate from current to target orientation"""
        if current == target:
            return self.STAY
        
        # Determine shorter rotation direction
        diff = (target - current) % 4
        if diff <= 2:
            return self.ROTATE_RIGHT
        else:
            return self.ROTATE_LEFT
    
    def _orientation_to_action(self, orientation):
        """Convert orientation to movement action"""
        if orientation == 0:  # Up
            return self.MOVE_UP
        elif orientation == 1:  # Left
            return self.MOVE_LEFT
        elif orientation == 2:  # Down
            return self.MOVE_DOWN
        elif orientation == 3:  # Right
            return self.MOVE_RIGHT
        return self.STAY
    
    def update_policy(self, actions, state, rewards):
        """No learning in this agent, but tracks steps for debugging"""
        pass