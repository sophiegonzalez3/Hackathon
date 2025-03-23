import numpy as np
import anthropic
import os
#from dotenv import load_dotenv

class MyAgent:
    def __init__(self, num_agents: int):
        # Load API key from environment or use the provided one
        #load_dotenv()
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "***********************************")
            
        self.num_agents = num_agents
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.conversation_history = []
        
        # Add state memory to detect loops
        self.previous_states = []
        self.previous_actions = []
        self.loop_detection_window = 10 

        # These are the keys the user will input (same as original)
        self.key_map = {
            'z': 'FORWARD',
            's': 'BACKWARD',
            'q': 'LEFT',
            'd': 'RIGHT',
            'v': 'ROTATE_RIGHT',
            'b': 'ROTATE_LEFT',
            'x': 'STAY'
        }
        
        # Initial system prompt
        self.system_prompt = self.system_prompt = """
You are an AI that controls agents in a navigation game. Make smart movement decisions to help agents reach their goals.

GOAL: Navigate through a maze to reach a target position while avoiding obstacles and walls.

AGENT CAPABILITIES:
- Move forward/backward/left/right relative to current orientation
- Rotate left/right to change orientation
- Use LIDAR sensors to detect obstacles

KEY RULES:
1. CRITICAL: Agent dies upon wall collision - NEVER move into a wall
2. CRITICAL: Agent dies if within 1 unit of a dynamic obstacle - maintain safe distance
3. The goal is to minimize distance to the target position

SENSOR INTERPRETATION:
- LIDAR readings show both distance and type of obstacle
- Type 0 = Clear path, Type 1 = Wall, Type 2 = Dynamic Obstacle, Type 3 = Another Agent
- If distance > 1, path is safe to move into
- ALWAYS check sensor readings before moving

NAVIGATION ALGORITHM:
1. First, scan surroundings (rotate to check all directions)
2. If front is clear AND moving forward reduces goal distance, move forward
3. If front is blocked OR moving forward increases goal distance:
   a. Rotate to find the clearest path toward goal
   b. If multiple clear paths, choose the one that most reduces goal distance
4. If stuck in a loop of repeated actions, try a different action

ORIENTATION GUIDE:
- UP (0): Moving forward decreases Y coordinate
- RIGHT (1): Moving forward increases X coordinate
- DOWN (2): Moving forward increases Y coordinate
- LEFT (3): Moving forward decreases X coordinate

MAZE PROPERTIES:
- 10x10 grid with walls
- Goals typically at opposite corners from starting position

For each agent, respond with ONLY ONE LETTER:
z: Forward, s: Backward, q: Left, d: Right, v: Rotate Right, b: Rotate Left, x: Stay

You are controlling an agent in a 10x10 maze. Build a mental map of the environment as you explore.

IMPORTANT STRATEGY FOR MAZE NAVIGATION:
1. Maintain a mental map of discovered maze walls and open spaces
2. When exploring, prioritize unexplored areas that bring you closer to the goal
3. If the direct path to goal is blocked, try the "right-hand rule" - keep your right hand on a wall and follow it
4. If stuck in a loop, deliberately try a direction you haven't explored yet
5. For efficient navigation:
   - First, try moving toward the goal if path is clear
   - If blocked, rotate to build map awareness
   - When choosing direction, prefer unexplored areas
   - Avoid backtracking unless necessary

CRITICAL SAFETY RULES:
- NEVER move toward a wall (type 1) at distance 1
- NEVER move toward a dynamic obstacle (type 2) at distance 1
- If front sensor shows distance 1 with obstacle, DO NOT move forward

For each agent, respond with ONLY ONE LETTER:
z: Forward, s: Backward, q: Left, d: Right, v: Rotate Right, b: Rotate Left, x: Stay

ou are an AI that controls agents in a navigation game. Your goal is to reach the target position safely.

CRITICAL SAFETY RULES:
1. NEVER move in a direction where a wall or obstacle is at distance 1.0
2. When radar shows 'Wall (dist: 1.0)' in any direction, DO NOT move in that direction
3. Pay extra attention to RIGHT and LEFT sensors - even if the front is clear

NAVIGATION STRATEGY:
1. Moving toward the goal is good, but NEVER at the cost of safety
2. When you see a wall ahead, ALWAYS rotate instead of trying to move forward
3. Use the loop detection warning - when stuck, try a completely different action
4. If you see walls on multiple sides, look for the clear path even if it takes you temporarily away from the goal
5. When you see "RIGHT: Wall (dist: 1.0)" do NOT use 'd' (right) command
6. When you see "LEFT: Wall (dist: 1.0)" do NOT use 'q' (left) command
7. If you're already facing RIGHT and the goal is to your right, but there's a wall at distance 1, ROTATE instead

ORIENTATION GUIDE:
- UP (0): z=forward moves UP (decreases Y)
- RIGHT (1): z=forward moves RIGHT (increases X)
- DOWN (2): z=forward moves DOWN (increases Y)
- LEFT (3): z=forward moves LEFT (decreases X)

SAFE MOVEMENT CHECKLIST:
- Before moving forward (z): Check FRONT distance > 1.0
- Before moving backward (s): Consider using rotation instead of backward
- Before moving left (q): Check LEFT distance > 1.0
- Before moving right (d): Check RIGHT distance > 1.0

For each agent, respond with ONLY ONE LETTER: z, s, q, d, v, b, or x

NO EXPLANATIONS - JUST THE LETTER!
"""
    
    def get_action(self, state: list, evaluation: bool = False):
        actions = []
        active_agents = []

        # Store current state for loop detection
        position_states = []
        for i in range(self.num_agents):
            if state[i][3] == 0:  # Only track active agents
                pos = tuple(state[i][0:2])
                orient = int(state[i][2])
                position_states.append((pos, orient))

        self.previous_states.append(position_states)
        if len(self.previous_states) > self.loop_detection_window:
            self.previous_states.pop(0)

        # Format the state information for each agent
        agent_states_text = ""

        for i in range(self.num_agents):
            agent_state = state[i]

            # Skip if agent is deactivated or evacuated
            if agent_state[3] > 0:
                print(f"\nAgent {i} has been {'evacuated' if agent_state[3] == 1 else 'deactivated'}! Skipping...")
                actions.append(0)  # Stay action for inactive agents
                continue

            active_agents.append(i)

            # Extract state information
            pos = agent_state[0:2]
            orient = int(agent_state[2])  # Current orientation (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
            orientation_map = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
            goal_pos = agent_state[4:6]
            goal_dist = np.linalg.norm(pos - goal_pos)

            # Calculate vector to goal
            goal_vector = goal_pos - pos

            # Direction vectors based on orientation
            dir_vectors = {
                0: np.array([0, -1]),  # UP
                1: np.array([1, 0]),   # RIGHT
                2: np.array([0, 1]),   # DOWN
                3: np.array([-1, 0])   # LEFT
            }

            # Calculate if moving forward would reduce goal distance
            current_dir = dir_vectors[orient]
            would_reduce_distance = np.dot(current_dir, goal_vector) > 0

            # LIDAR data
            front_dist = agent_state[6]
            front_type = agent_state[7]
            right_dist = agent_state[8]
            right_type = agent_state[9]
            left_dist = agent_state[10]
            left_type = agent_state[11]

            obstacle_map = {0: "Clear", 1: "Wall", 2: "Dynamic Obstacle", 3: "Another Agent"}

            # Check for potential loop
            loop_detected = False
            if len(self.previous_states) >= 5 and len(self.previous_actions) >= 5:
                # Check if position has been revisited multiple times
                position_count = sum(1 for prev_state in self.previous_states[-5:] 
                                   if len(prev_state) > i and prev_state[i][0] == tuple(pos))
                if position_count >= 3:
                    loop_detected = True

                # Check if same action has been repeated
                last_actions = [act[i] if len(act) > i else None for act in self.previous_actions[-5:]]
                if len(set(last_actions)) == 1 and None not in last_actions:
                    loop_detected = True

            # Format the state for this agent
            agent_states_text += f"""
            AGENT {i} STATUS:
            Position: {pos}, Facing: {orientation_map.get(orient, 'UNKNOWN')}
            Goal: {goal_pos}, Distance to goal: {goal_dist:.2f}
            Moving forward would {"reduce" if would_reduce_distance else "increase"} distance to goal

            RADAR READINGS (relative to agent orientation):
              FRONT: {obstacle_map.get(int(front_type), 'Unknown')} (dist: {front_dist:.1f})
              RIGHT: {obstacle_map.get(int(right_type), 'Unknown')} (dist: {right_dist:.1f})
              LEFT:  {obstacle_map.get(int(left_type), 'Unknown')} (dist: {left_dist:.1f})
            """

            if loop_detected:
                agent_states_text += """
            WARNING: You may be stuck in a loop! Try a different action than recent ones.
            """
        
        if not active_agents:
            print("No active agents found.")
            return [0] * self.num_agents
            
        agent_states_text += "\n\nFor each agent, respond with a single letter move command (one per line)."
            
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": agent_states_text})
        
        # Print status for debugging
        print(f"\nSending state data for {len(active_agents)} active agents to Claude API")
        
        print(agent_states_text)
        
        # Send to Claude API
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                system=self.system_prompt,
                messages=self.conversation_history,
                max_tokens=100,
                temperature=0.1  # Lower temperature for more consistent responses
            )
            
            # Get Claude's response
            claude_response = response.content[0].text.strip()
            self.conversation_history.append({"role": "assistant", "content": claude_response})
            
            print(f"Claude response: {claude_response}")
            
            # Process response for each agent
            response_lines = claude_response.strip().split('\n')
            response_chars = [line.strip().lower()[0] if line.strip() else 'x' for line in response_lines]
            
            # Make sure we have valid commands
            response_chars = [c for c in response_chars if c in self.key_map]
            
            # Convert Claude's letter responses to actions
            for idx, agent_idx in enumerate(active_agents):
                if idx < len(response_chars):
                    command_key = response_chars[idx]
                    command = self.key_map.get(command_key, 'STAY')
                    
                    # Convert relative command to appropriate action based on orientation
                    orient = int(state[agent_idx][2])
                    
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
                else:
                    action = 0  # Default to STAY if Claude didn't provide enough responses
                
                # Store the action
                while len(actions) <= agent_idx:
                    actions.append(0)
                actions[agent_idx] = action
                
                # Print for debugging - similar to the original agent output
                action_map = {0: "STAY", 1: "MOVE UP", 2: "MOVE DOWN", 3: "MOVE LEFT", 
                              4: "MOVE RIGHT", 5: "ROTATE RIGHT", 6: "ROTATE LEFT"}
                print(f"Command: {command} → Action: {action_map.get(action)}")
                
        except Exception as e:
            print(f"Error communicating with Claude API: {e}")
            # Default to STAY for all agents in case of API error
            actions = [0] * self.num_agents
        
        # Make sure we have actions for all agents
        while len(actions) < self.num_agents:
            actions.append(0)
        
        self.previous_actions.append(actions.copy())
        if len(self.previous_actions) > self.loop_detection_window:
            self.previous_actions.pop(0)
            
        return actions
    
    def update_policy(self, actions: list, state: list, reward):
        # Calculate total reward for this step (handling both float and array types)
        total_reward = np.sum(reward) if isinstance(reward, (list, np.ndarray)) else reward

        # Add the reward information to the conversation history
        reward_message = f"The last actions received a reward of: {total_reward:.2f}"
        self.conversation_history.append({"role": "user", "content": reward_message})

        # Keep conversation history manageable (last 10 exchanges)
        if len(self.conversation_history) > 20:
            # Keep last 10 exchanges (20 messages - user and assistant alternating)
            self.conversation_history = self.conversation_history[-20:]

        print(f"Updated Claude with reward: {total_reward:.2f}")