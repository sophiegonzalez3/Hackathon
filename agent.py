import numpy as np

class MyAgent:
    def __init__(self, num_agents: int):        
        self.rng = np.random.default_rng()
        self.num_agents = num_agents
        # These are the keys the user will input
        self.key_map = {
            'z': 'FORWARD',
            's': 'BACKWARD',
            'q': 'LEFT',
            'd': 'RIGHT',
            'v': 'ROTATE_RIGHT',
            'b': 'ROTATE_LEFT',
            'x': 'STAY'
        }
    
    def get_action(self, state: list, evaluation: bool = False):
        actions = []
        
        for i in range(self.num_agents):
            agent_state = state[i]
            
            # Skip if agent is deactivated or evacuated
            if agent_state[3] > 0:
                if agent_state[3] == 1:
                    print(f"\nAgent {i} has been evacuated! Skipping...")
                else:
                    print(f"\nAgent {i} has been deactivated! Skipping...")
                actions.append(0)
                continue
            
            # Extract state information
            pos = agent_state[0:2]
            orient = int(agent_state[2])  # Current orientation (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
            goal_pos = agent_state[4:6]
            goal_dist = np.linalg.norm(pos - goal_pos)
            
            # LIDAR data
            front_dist = agent_state[6]
            front_type = agent_state[7]
            right_dist = agent_state[8]
            right_type = agent_state[9]
            left_dist = agent_state[10]
            left_type = agent_state[11]
 
            # Display agent status
            orientation_map = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
            obstacle_map = {0: "Clear", 1: "Wall", 2: "Dynamic Obstacle", 3: "Another Agent"}
            
            print(f"Absolute position \n")
            print(f"Position: {pos}")
            print(f"Orientation: {orient}")
            print(f"Goal Position: {goal_pos}")
            print(f"Distance to Goal: {goal_dist:.2f}")
            print(f"Radar - Front Distance: {front_dist}, Type: {front_type} - {obstacle_map.get(int(front_type), 'Unknown')} " )
            print(f"Radar - Right Distance: {right_dist}, Type: {right_type} - {obstacle_map.get(int(right_type), 'Unknown')} ")
            print(f"Radar - Left Distance: {left_dist}, Type: {left_type} - {obstacle_map.get(int(left_type), 'Unknown')}")

                 
            print("\n" + "="*50)
            print(f"AGENT {i} STATUS:")
            print(f"Position: {pos}, Facing: {orientation_map.get(orient, 'UNKNOWN')}")
            print(f"Goal: {goal_pos}, Distance to goal: {goal_dist:.2f}")
            print("\nRADAR READINGS (relative to agent orientation):")
            print(f"  FRONT: {obstacle_map.get(int(front_type), 'Unknown')} (dist: {front_dist:.1f})")
            print(f"  RIGHT: {obstacle_map.get(int(right_type), 'Unknown')} (dist: {right_dist:.1f})")
            print(f"  LEFT:  {obstacle_map.get(int(left_type), 'Unknown')} (dist: {left_dist:.1f})")
            
            # Display controls
            print("\nCONTROLS (RELATIVE TO AGENT ORIENTATION):")
            print("  [Z] Forward     [S] Backward")
            print("  [Q] Left        [D] Right")
            print("  [V] Rotate R    [B] Rotate L")
            print("  [X] Stay Still")
            
            # Get user input
            valid_input = False
            while not valid_input:
                user_input = input("\nEnter command: ").lower()
                if user_input in self.key_map:
                    command = self.key_map[user_input]
                    valid_input = True
                else:
                    print("Invalid command. Use Z/S/Q/D/V/B/X.")
            
            # Convert relative command to appropriate action based on orientation
            if command == "STAY":
                action = 0
            elif command == "ROTATE_RIGHT":
                action = 5
            elif command == "ROTATE_LEFT":
                action = 6
            else:
                # Movement mapping based on orientation
                movement_map = {
                    0: {"FORWARD": 1, "BACKWARD": 2, "LEFT": 3, "RIGHT": 4},  # UP
                    1: {"FORWARD": 4, "BACKWARD": 3, "LEFT": 1, "RIGHT": 2},  # RIGHT
                    2: {"FORWARD": 2, "BACKWARD": 1, "LEFT": 4, "RIGHT": 3},  # DOWN
                    3: {"FORWARD": 3, "BACKWARD": 4, "LEFT": 2, "RIGHT": 1}   # LEFT
                }
                action = movement_map[orient][command]
            
            # Display the selected action
            action_map = {0: "STAY", 1: "MOVE UP", 2: "MOVE DOWN", 3: "MOVE LEFT", 
                          4: "MOVE RIGHT", 5: "ROTATE RIGHT", 6: "ROTATE LEFT"}
            print(f"Command: {command} → Action: {action_map.get(action)}")
            
            actions.append(action)
        
        return actions
    
    def update_policy(self, actions: list, state: list, reward: float):
        pass