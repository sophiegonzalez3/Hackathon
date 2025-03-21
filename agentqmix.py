import copy
from itertools import pairwise
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn, optim
from typing import List, Tuple, Dict, Set
from buffer import SequenceBuffer
import traceback

def save_tensor_as_markdown(tensor, filename="matrix.md"):
    """
    Saves a PyTorch tensor as a markdown-formatted table to a file.

    Args:
        tensor (torch.Tensor): The PyTorch tensor to be saved.
        filename (str): The name of the file to save the markdown table (default is 'matrix.md').
    """
    # Convert the tensor to a NumPy array
    matrix = tensor.cpu().numpy()

    # Start building the markdown table
    markdown_table = "| " + " | ".join([f"Col {i+1}" for i in range(matrix.shape[1])]) + " |\n"
    markdown_table += "|" + " --- |" * matrix.shape[1] + "\n"

    # Add rows of the matrix
    for row in matrix:
        markdown_table += "| " + " | ".join(map(str, row)) + " |\n"

    # Write to a .txt or .md file
    with open(filename, "w") as f:
        f.write(markdown_table)




def central_state_static(
    #agent_states: torch.Tensor,
    grid_size: int,
    walls: Set[Tuple[int, int]],
    goal_area: List[Tuple[int, int]],
    optimal_paths: List[Tuple[int, int]], 
    device: torch.device
) -> torch.Tensor:
    """
    Creates an enhanced central state for QMIX with separate matrices for static and dynamic elements.
    
    Args:
        agent_states: Individual agent states tensor [batch_size, num_agents, state_size]
        grid_size: Size of the environment grid
        walls: Set of wall coordinates (x, y)
        goal_area: List of goal area coordinates (x, y)
        optimal_paths: List of coordinates (x, y) representing optimal paths from A*
        device: Torch device
        
    Returns:
        Enhanced central state tensor [batch_size, 2, grid_size, grid_size]
    """
    
    
    # Fill the static environment matrix (channel 0)
    MAX_GRID_SIZE = 30
    needs_padding = grid_size < MAX_GRID_SIZE
    print("\nGoal area : " ,goal_area)

    static_matrix = torch.ones((grid_size, grid_size), device=device)  # Default: +1 for neutral
    
    # Add walls: -5
    for wall in walls:
        
        static_matrix[wall] = -5
    
    # Add goals: +100
    for goal in goal_area:
        static_matrix[goal] = 100
    
    # Add optimal paths: +2
    for path_point in optimal_paths:
        # Only mark if not already a wall or goal
        if static_matrix[path_point] == 1:
            static_matrix[path_point] = 2

    #save_tensor_as_markdown(static_matrix, "staticBefore.md")
    # Detect the corner where the goal is currently located
    goal_center_x = goal_area[0][0]
    goal_center_y = goal_area[0][1]
    #print("goal center : " , goal_center_x, goal_center_y)
        
    # Determine which corner the goal is closest to
    is_bottom = goal_center_x >= grid_size / 2
    is_right = goal_center_y >= grid_size / 2
    
    # Rotate the matrix based on goal position to ensure goal is bottom right
    if not is_right and not is_bottom:  # Goal is in top-left, rotate 180°
        print("Static Goal is in top-left, rotate 180°\n")
        static_matrix = torch.flip(static_matrix, [0, 1])
    elif not is_right and is_bottom:  # Goal is in bottom-left, rotate 90° counter clockwise
        static_matrix = torch.rot90(static_matrix, k=1, dims=[0, 1])
        print("Static Goal is in bottom-left, rotate 90° counter-clockwise\n")
    elif is_right and not is_bottom:  # Goal is in top-right, rotate 90° clockwise
        static_matrix = torch.rot90(static_matrix, k=-1, dims=[0, 1])
        print("Static Goal is in top-right, rotate 90° clockwise\n")
    else :
        print("Static bottom-righ\n")
    # If goal is already bottom-right, no rotation needed
    
    if needs_padding:
        # Create a matrix with the target size, filled with wall values (-5)
        padded_matrix = torch.full((MAX_GRID_SIZE, MAX_GRID_SIZE), -5, device=device)
        
        # Place the actual grid in the top-left corner
        padded_matrix[:grid_size, :grid_size] = static_matrix
        
        # Use the padded matrix
        static_matrix = padded_matrix
    
    # Optional: save for visualization
    #save_tensor_as_markdown(static_matrix, "staticAfter.md")
    
    return static_matrix

 

def central_state_dynamic(
    agent_states: torch.Tensor,
    grid_size: int,
    agent_positions: List[np.ndarray],
    dynamic_obstacles: List[Tuple[int, int]],
    evacuated_agents: Set[int],
    deactivated_agents: Set[int],
    comm_range: int,  # Added parameter for communication range
    goal_area: List[Tuple[int, int]],
    device: torch.device
) -> torch.Tensor:
    """
    Creates the dynamic part of the enhanced central state for QMIX.
    Ensures the output tensor is always 30x30 by adding padding if the grid_size is smaller.
    
    Includes:
    - Halos around obstacles (-10 for adjacent cells)
    - Graduated positive halos around agents (from -5 at center to +5 at range extremity)
    
    Args:
        agent_states: Individual agent states tensor [batch_size, num_agents, state_size]
        grid_size: Size of the environment grid (can range from 10 to 30)
        agent_positions: List of agent positions as numpy arrays
        dynamic_obstacles: List of dynamic obstacle positions (x, y)
        evacuated_agents: Set of indices of evacuated agents
        deactivated_agents: Set of indices of deactivated agents
        comm_range: Communication range for agents (Manhattan distance)
        device: Torch device
        
    Returns:
        Enhanced dynamic central state tensor [grid_size, grid_size] or [30, 30] if padded
    """
    # Maximum grid size we want to support
    MAX_GRID_SIZE = 30
    
    # Check if we need padding
    needs_padding = grid_size < MAX_GRID_SIZE
    
    # Create the appropriate sized matrix for the actual grid
    dynamic_matrix = torch.ones((grid_size, grid_size), device=device)  # Default: +1 for neutral
    
    # Add dynamic obstacles: -10
    for obstacle in dynamic_obstacles:
        x, y = obstacle
        dynamic_matrix[x, y] = -10
        
        # Add negative halo around obstacles
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                # Skip the center (obstacle itself)
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy
                # Check bounds
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    # Only overwrite if the cell is not already an obstacle
                    if dynamic_matrix[nx, ny] != -10:
                        dynamic_matrix[nx, ny] = -10
    
    # Add agents: -5 and graduated positive halos
    for i, pos in enumerate(agent_positions):
        if i not in evacuated_agents and i not in deactivated_agents:
            x, y = int(pos[0]), int(pos[1])
            
            # Mark agent position
            dynamic_matrix[x, y] = -5
            
            # Create graduated positive halo around agent
            for radius in range(1, comm_range + 1):
                # Calculate graduated value: -5 at center to +5 at extremity
                # Linear interpolation
                halo_value = -5 + (10 * radius / comm_range)
                
                # Iterate through all cells at Manhattan distance = radius
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        # Check if the point is exactly at Manhattan distance = radius
                        if abs(dx) + abs(dy) == radius:
                            nx, ny = x + dx, y + dy
                            # Check bounds
                            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                                # Prioritize: Keep obstacle values, but overwrite neutral values
                                if dynamic_matrix[nx, ny] != -10 and dynamic_matrix[nx, ny] != -5:
                                    dynamic_matrix[nx, ny] = halo_value
    
    # Detect the corner where the goal is currently located
    goal_center_x = goal_area[0][0]
    goal_center_y = goal_area[0][1]
    #print("goal center : " , goal_center_x, goal_center_y)
        
    # Determine which corner the goal is closest to
    is_bottom = goal_center_x >= grid_size / 2
    is_right = goal_center_y >= grid_size / 2
    
    # Rotate the matrix based on goal position to ensure goal is bottom right
    if not is_right and not is_bottom:  # Goal is in top-left, rotate 180°
        #print("\nGoal is in top-left, rotate 180°")
        dynamic_matrix = torch.flip(dynamic_matrix, [0, 1])
    elif not is_right and is_bottom:  # Goal is in bottom-left, rotate 90° counter clockwise
        dynamic_matrix = torch.rot90(dynamic_matrix, k=1, dims=[0, 1])
        #print("\nGoal is in bottom-left, rotate 90° counter clockwise")
    elif is_right and not is_bottom:  # Goal is in top-right, rotate 90° clockwise
        dynamic_matrix = torch.rot90(dynamic_matrix, k=-1, dims=[0, 1])
        #print("\nGoal is in top-right, rotate 90° clockwise")
    else : # If goal is already bottom-right, on rotate pas
        #print("bottom-righ")
        pass
    
    
    if needs_padding:
        # Create a matrix with the target size, filled with wall values (-5)
        padded_matrix = torch.full((MAX_GRID_SIZE, MAX_GRID_SIZE), -5, device=device)
        
        # Place the actual grid in the top-left corner
        padded_matrix[:grid_size, :grid_size] = dynamic_matrix
        
        # Use the padded matrix
        dynamic_matrix = padded_matrix
    
    # Optional: save for visualization
    #save_tensor_as_markdown(dynamic_matrix, "dynamic.md")
    return dynamic_matrix



def state_to_tensor(state: list, device: torch.device) -> torch.Tensor:
    """Convert state from list of arrays to tensor."""
    agg_state = np.vstack(state)
    return torch.from_numpy(agg_state).to(device)


def action_to_array(actions: torch.Tensor) -> npt.NDArray[np.int64]:
    """Convert actions from torch tensor to numpy array."""
    return actions.detach().cpu().numpy().tolist()


def make_dnn(layer_sizes: list[int]) -> nn.Sequential:
    """Build a MLP with ReLU activation functions."""
    layers: list[nn.Module] = []
    for h_in, h_out in pairwise(layer_sizes[:-1]):
        layers.append(nn.Linear(h_in, h_out))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
    return nn.Sequential(*layers)


def soft_update(model: nn.Module, target_model: nn.Module, tau: float) -> None:
    """Soft update model parameters: θ_target = τ*θ_local + (1-τ)*θ_target"""
    for param, target_param in zip(
        model.parameters(), target_model.parameters(), strict=True
    ):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class StateProcessor(nn.Module):
    """Processes the central state with CNNs to extract spatial features."""
    def __init__(self, grid_size=30, channels=2, feature_dim=128):
        super().__init__()
        self.grid_size = grid_size
        self.channels = channels
        
        # CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 15x15
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3))  # 3x3
        )
        
        # Flatten and project
        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, feature_dim),
            nn.ReLU()
        )
        
    def forward(self, state_matrix, agent_states):
        """
        Args:
            state_matrix: [batch_size, channels, grid_size, grid_size]
            agent_states: [batch_size, num_agents, state_size]
        """
        batch_size = state_matrix.shape[0]
        
        # Process grid with CNN
        spatial_features = self.conv_layers(state_matrix)
        spatial_features = spatial_features.view(batch_size, -1)
        spatial_features = self.fc(spatial_features)
        
        # Flatten agent states
        agent_features = agent_states.view(batch_size, -1)
        
        # Concatenate spatial and agent features
        combined_features = torch.cat([spatial_features, agent_features], dim=1)
        
        return combined_features


class QNetRNN(nn.Module):
    """Q-Network with RNN to maintain memory of past states."""
    def __init__(self, state_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        
        # Initial feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # GRU layer for memory
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1  # Add dropout for regularization
        )
        
        # Action predictor with attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.action_head = nn.Linear(hidden_size, action_size)
        
        # Initialize hidden state
        self.hidden = None
        
        # Initialize the discovered grid representation
        # 0: unknown, 1: empty, 2: wall, 3: dynamic obstacle, 4: agent
        
        
        
        # Keep track of visited cells
        self.visited_cells = set()

    def update_grid(self, state):
        
        grid_size = max(state)
        
        """Update the internal grid representation based on LIDAR data"""
        # Extract relevant information from state
        x, y = int(state[0]), int(state[1])
        goal_pos = state[4:6]
        goal_dist = np.linalg.norm((x,y) - goal_pos)
        orientation = int(state[2]) # Current orientation (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
        orientation_map = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
        obstacle_map = {0: "Clear", 1: "Wall", 2: "Dynamic Obstacle", 3: "Another Agent"}
        front_dist, front_type = state[6], state[7]
        right_dist, right_type = state[8], state[9]
        left_dist, left_type = state[10], state[11]
        
        # Mark current position as visited and empty
        self.visited_cells.add((x, y))
        self.discovered_grid[x, y] = 1
        
        # Movement mapping based on orientation
        movement_map = {
            0: {"FORWARD": 1, "BACKWARD": 2, "LEFT": 3, "RIGHT": 4},  # UP
            1: {"FORWARD": 4, "BACKWARD": 3, "LEFT": 1, "RIGHT": 2},  # RIGHT
            2: {"FORWARD": 2, "BACKWARD": 1, "LEFT": 4, "RIGHT": 3},  # DOWN
            3: {"FORWARD": 3, "BACKWARD": 4, "LEFT": 2, "RIGHT": 1}   # LEFT
        }

        # with open('output.txt', 'w') as file:
        print("\n" + "="*50)
        print("deduc grid size", grid_size)
        print(f"AGENT {i} STATUS:")
        print(f"Position: {(x,y)}, Facing: {orientation_map.get(orientation, 'UNKNOWN')}")
        print(f"Goal: {goal_pos}, Distance to goal: {goal_dist:.2f}")
        print("\nRADAR READINGS (relative to agent orientation):")
        print(f"  FRONT: {obstacle_map.get(int(front_type), 'Unknown')} (dist: {front_dist:.1f})")
        print(f"  RIGHT: {obstacle_map.get(int(right_type), 'Unknown')} (dist: {right_dist:.1f})")
        print(f"  LEFT:  {obstacle_map.get(int(left_type), 'Unknown')} (dist: {left_dist:.1f})")

        
    def reset_hidden(self, batch_size=1, device=None):
        """Reset hidden state at the beginning of an episode."""
        self.hidden = torch.zeros(
            self.gru.num_layers, 
            batch_size, 
            self.gru.hidden_size, 
            device=device
        )
        
    def forward(self, state, sequence_length=1):
        """
        Forward pass through the network.
        
        Args:
            state: [batch_size, state_size] or [state_size]
            sequence_length: Length of sequence for reshaping (default=1 for single step)
            
        Returns:
            Q-values for each action
        """
        # Handle both batched and single inputs
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single_input = True
        else:
            single_input = False
            
        batch_size = state.size(0)
        device = state.device
        
        # Reset hidden state if needed
        if self.hidden is None or self.hidden.size(1) != batch_size:
            self.reset_hidden(batch_size, device)
            
        # Extract features
        x = self.feature_net(state)
        
                # Add sequence dimension if processing single steps
        if sequence_length == 1:
            x = x.unsqueeze(1)
        
        # Process through GRU
        gru_out, self.hidden = self.gru(x, self.hidden)
        
        # Apply attention for better temporal focus (if sequence_length > 1)
        if sequence_length > 1:
            # Simple attention mechanism
            attn_weights = self.attention(gru_out).softmax(dim=1)
            x = torch.sum(gru_out * attn_weights, dim=1)
        else:
            x = gru_out.squeeze(1)
        
        # Predict Q-values
        q_values = self.action_head(x)
        
        # Remove batch dimension if input was single state
        if single_input:
            q_values = q_values.squeeze(0)
            
        return q_values
        
    def get_hidden_state(self):
        """Return the current hidden state (detached to avoid backprop issues)."""
        return self.hidden.detach() if self.hidden is not None else None
        
    def set_hidden_state(self, hidden):
        """Set the hidden state."""
        self.hidden = hidden


class CNNMixingNetwork(nn.Module):
    """QMIX Mixing Network with CNN for processing spatial state."""
    
    def __init__(
        self, num_agents: int, grid_size: int, agent_state_size: int, mixing_hidden_dim: int
    ) -> None:
        super().__init__()
        
        self.num_agents = num_agents
        self.mixing_hidden_dim = mixing_hidden_dim  # Store as instance attribute
        self.agent_state_size = agent_state_size
        self.grid_size = grid_size
        
        # Calculate CNN output size based on grid_size
        # For a 30x30 grid with two max pooling layers (each dividing by 2)
        # The spatial dimensions will be reduced to 7x7
        self.cnn_output_size = 32 * 7 * 7
        
        # Simplified CNN for processing the grid state
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 15x15
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.cnn_output_size, mixing_hidden_dim),
            nn.ReLU()
        )
        
        # Calculate the combined feature size
        processed_state_size = mixing_hidden_dim + (num_agents * agent_state_size)
        
        # Hypernetworks for the mixing network
        # First layer weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(processed_state_size, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, num_agents * mixing_hidden_dim),
        )

        # First layer bias
        self.hyper_b1 = nn.Sequential(
            nn.Linear(processed_state_size, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, mixing_hidden_dim),
        )

        # Second layer weights
        self.hyper_w2 = nn.Sequential(
            nn.Linear(processed_state_size, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, mixing_hidden_dim),
        )

        # Second layer bias
        self.hyper_b2 = nn.Sequential(
            nn.Linear(processed_state_size, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, state_matrix: torch.Tensor, agent_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_qs: Individual agent Q-values [batch_size, num_agents]
            state_matrix: Environmental state [batch_size, channels, grid_size, grid_size]
            agent_states: Individual agent states [batch_size, num_agents, state_size]
        Returns:
            q_tot: Mixed Q-value [batch_size, 1]
        """
        try:
            batch_size = agent_qs.size(0)
            
            # Process the grid state using CNN
            grid_features = self.conv_net(state_matrix)
            
            # Flatten agent states
            agent_features = agent_states.view(batch_size, -1)
            
            # Combine features
            combined_features = torch.cat([grid_features, agent_features], dim=1)
            
            # Generate mixing network weights and biases
            w1 = torch.abs(self.hyper_w1(combined_features)).view(
                batch_size, self.num_agents, self.mixing_hidden_dim
            )
            b1 = self.hyper_b1(combined_features).view(batch_size, 1, self.mixing_hidden_dim)

            w2 = torch.abs(self.hyper_w2(combined_features))
            w2 = w2.view(batch_size, self.mixing_hidden_dim, 1)
            b2 = self.hyper_b2(combined_features).view(batch_size, 1, 1)

            # Forward pass through the mixing network
            hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
            q_tot = torch.bmm(hidden, w2) + b2

            return q_tot.squeeze(2)
        except Exception as e:
            print(f"Error in mixer forward: {e}")
            print(f"Agent Qs shape: {agent_qs.shape}")
            print(f"State matrix shape: {state_matrix.shape}")
            print(f"Agent states shape: {agent_states.shape}")
            raise




class MyAgent:
    def __init__(
        self,
        num_agents: int,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Agent configuration
        self.rng = np.random.default_rng()
        self.num_agents = num_agents
        self.action_low = 0
        self.action_high = 6  # 7 actions (0-6)
        self.max_grid_size = 30

        # State and action dimensions
        self.state_size = 10 * num_agents + 2  # As in your original code
        self.action_size = self.action_high + 1
        
        # Network sizes
        self.q_net_hidden_size = 128
        self.mixing_hidden_size = 128

        # Learning parameters
        self.buffer_size = 50_000  # Increased buffer size for better experience replay
        self.batch_size = 128  # Smaller batch size for more frequent updates
        self.lr = 1e-4  # Smaller learning rate for stability
        self.gamma = 0.99
        self.tau = 0.01  # Even slower target updates for more stability
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # Higher minimum for continued exploration
        self.epsilon_decay = 0.995  # Slower decay
        
        # Calculate the central state size correctly
        # Just the flattened grid (30x30) from extract_central_state
        self.central_state_size = self.max_grid_size * self.max_grid_size
        self.static_matrix = torch.ones((self.max_grid_size, self.max_grid_size), device=self.device)
        
        # Buffer for experience replay
        self.sequence_length = 8
        self.buffer = SequenceBuffer(
            self.buffer_size,
            self.num_agents,
            self.state_size,
            self.central_state_size,
            self.sequence_length,
            self.device
        )
        
        # Initialize performance tracking attributes
        self.rewards_history = []
        self.loss_history = []

        # Q-networks with RNN for memory
        self.q_net = QNetRNN(self.state_size, self.action_size, self.q_net_hidden_size).to(self.device)
        
        # CNN Mixer for better spatial reasoning
        self.mixer = CNNMixingNetwork(
            self.num_agents, self.max_grid_size, self.state_size, self.mixing_hidden_size
        ).to(self.device)

        # self.reset_networks()

        self.target_q_net = copy.deepcopy(self.q_net)
        self.target_q_net.eval()
        self.target_mixer = copy.deepcopy(self.mixer)
        self.target_mixer.eval()
        
        print(f"Networks initialized - Q-Net and Mixer created")

        self.optimizer = optim.Adam(
            list(self.q_net.parameters()) + list(self.mixer.parameters()), self.lr,  weight_decay=1e-5
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5,
            patience=10,
            verbose=True
        )

        
        # Episode tracking
        self.episode_count = 0
        self.current_episode = 0
        self.hidden_states =  {}  # Dictionary to store agent hidden states by episode and agent ID

    def extract_central_state(self, states: torch.Tensor, env, max_grid_size:int,  device) -> torch.Tensor:
        batch_size = 1
        central_state = torch.zeros((batch_size, 1, max_grid_size, max_grid_size), device=device)
        dynamic_matrix = central_state_dynamic( states, env.grid_size, env.agent_positions, env.dynamic_obstacles, env.evacuated_agents, env.deactivated_agents, env.communication_range, env.goal_area, device)
        static_matrix = self.static_matrix.float()
        # Merging both dyanmic and static
        merged_matrix = torch.zeros_like(dynamic_matrix, device=device).float()
        # Create masks for different conditions
        both_positive = (dynamic_matrix > 0) & (static_matrix > 0)
        save_tensor_as_markdown(both_positive, "both_positive.md")
        #print(both_positive)
        other_cases = ~both_positive  # Everything else
        
        # Apply the merging rules
        merged_matrix[both_positive] = dynamic_matrix[both_positive] + static_matrix[both_positive]
        merged_matrix[other_cases] = torch.min(dynamic_matrix[other_cases], static_matrix[other_cases])
        
        # Assign to central state and return
        central_state[:, 0] = merged_matrix
        save_tensor_as_markdown(merged_matrix, "merged.md")
        return central_state.flatten()
#    
    def reset_episode(self, env, episode_reward=None):
        """Reset episode-specific variables like RNN hidden states."""
        self.episode_count += 1
        self.current_episode += 1
        self.track_weight_changes()
        self.buffer.start_new_episode()
        self.static_matrix = central_state_static(env.grid_size, env.walls, env.goal_area, env.optimal_path, self.device)
        
        # Initialize empty hidden states dictionary for this episode
        self.hidden_states[self.current_episode] = {}
        for i in range(self.num_agents):
            self.q_net.reset_hidden(device=self.device)
            self.hidden_states[self.current_episode][i] = self.q_net.get_hidden_state()

        # Clean up old episodes to avoid memory leaks
        if self.current_episode > 5:  # Keep only recent episodes
            if self.current_episode - 5 in self.hidden_states:
                del self.hidden_states[self.current_episode - 5]

        # Step the learning rate scheduler if it exists
        if hasattr(self, 'scheduler') and episode_reward is not None:
            #self.scheduler.step()
            self.scheduler.step(episode_reward)
        
        # Anneal epsilon if needed
        if self.episode_count % 1 == 0:  # Adjust epsilon after every episode
            self.update_epsilon()
            
    def get_action(self, state: list, evaluation: bool = False):
        """Select actions for each agent."""
        actions = []
        state = self.rotate_state(state)
        for agent, agent_state in enumerate(state):
            # Retrieve this agent's hidden state for the current episode
            if self.current_episode not in self.hidden_states:
                self.hidden_states[self.current_episode] = {}
            agent_hidden = self.hidden_states[self.current_episode].get(agent)

            # Exploration (random action)
            if (not evaluation) and self.rng.random() < self.epsilon:
                actions.append(self.rng.integers(self.action_low, self.action_high))
                continue

            else:
                try:
                    # Set the hidden state for this specific agent
                    if agent_hidden is not None:
                        self.q_net.set_hidden_state(agent_hidden)
                    else:
                        # Initialize if not exists
                        self.q_net.reset_hidden(device=self.device)
                    
                    # Convert numpy array to tensor and ensure proper dimensioning
                    agent_state_tensor = torch.from_numpy(agent_state).float().to(self.device)
                    
                    # Forward pass through Q-network - detach to prevent gradients
                    with torch.no_grad():
                        q_values = self.q_net(agent_state_tensor)
                    
                    # Select the best action
                    if not evaluation and self.rng.random() < 0.05:  # 5% chance of random variation
                        # Get values as numpy for selection
                        q_values_np = q_values.cpu().detach().numpy()
                        top_actions = np.argsort(q_values_np)[-3:]  # Get indices of top 3 actions
                        a = self.rng.choice(top_actions)  # Randomly select one of the top actions
                    else:
                        a = torch.argmax(q_values).item()  # Choose the best action
                    
                    actions.append(a)
                    
                    # Save the updated hidden state
                    self.hidden_states[self.current_episode][agent] = self.q_net.get_hidden_state()
                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"Error in get_action for agent {agent}: {e}")
                    print(f"Error details:\n{tb}")
                    # Fallback to random action in case of error
                    actions.append(self.rng.integers(self.action_low, self.action_high))
                
        return actions


    def update_policy(
        self,
        actions: list,
        state: list,
        reward: list,
        next_state: list,
        done: bool,
        env
    ):
        """Update policy based on experience."""
        # Convert state and prepare central state representation
        state = self.rotate_state(state)
        state_tensor = state_to_tensor(state, self.device)
        central_state = self.extract_central_state(state, env, self.max_grid_size, self.device)
        
        next_state_tensor = state_to_tensor(next_state, self.device)
        next_central_state = self.extract_central_state(next_state, env, self.max_grid_size, self.device)
        
        rewards_tensor = torch.tensor(reward, device=self.device)
        
        # Store in buffer
        self.buffer.append(
            state=state_tensor,
            central_state=central_state,
            action=torch.tensor(actions, device=self.device),
            reward=rewards_tensor,
            next_state=next_state_tensor,
            next_central_state=next_central_state,
            done=torch.tensor(int(done), device=self.device),
        )

        # Update networks if enough samples
        if len(self.buffer) >= self.batch_size:
            self.train_step()
            
        # Reset episode if done
        if done:
            self.reset_episode(env, sum(reward))

    def _shape_rewards(self, rewards, state, next_state, env):
        """Use the existing reward function - no additional shaping needed."""
        # Your existing compute_reward function already handles sophisticated reward shaping
        # This method is mainly here for extension if needed
        return rewards
    def rotate_state(self, state:list):
        # Determine which corner the goal is closest to (only need first agent data for it)
        agent_0 = state[0]
        grid_size = np.max(state.flatten())  ########## WARNING je suppose que c'est vrai et en plus que pour le goal pos est toujours la case la plus eloigne
        # print("State len for agent 0 : " , len(state[0]))
        # print("Deducted grid size : ", grid_size +1 ) 
        # print("initial state : " , state)
        goal_pos = agent_0[4:6]
        goal_center_x = goal_pos[0]
        goal_center_y = goal_pos[1]
        is_bottom = goal_center_x >= (grid_size+1) / 2
        is_right = goal_center_y >= (grid_size+1)/ 2
        complete_new_state = np.empty(4, dtype=object)
        for agent_idx in range(self.num_agents) : 
            agent_state = state[agent_idx]
            if agent_state[0] == -1 or  agent_state[1] ==-1 : #Agent mort
                #print("agent mort")
                complete_new_state[agent_idx] = agent_state
                continue
            
            # Extract components from the agent state
            x, y = agent_state[0], agent_state[1]
            orientation = agent_state[2]
            status = agent_state[3]
            goal_x, goal_y = agent_state[4], agent_state[5]
            

            # Rotate the state based on goal position to ensure goal is bottom right
            if not is_right and not is_bottom:  # Goal is in top-left, rotate 180°
                #print("\nGoal is in top-left, rotate 180°")
                # Flip coordinates
                new_x = grid_size - x
                new_y = grid_size - y
                new_goal_x = grid_size - goal_x
                new_goal_y = grid_size - goal_y
                # Adjust orientation (rotate 180°)
                new_orientation = (orientation + 2) % 4
                
            elif not is_right and is_bottom:  # Goal is in bottom-left, rotate 90° counter clockwise
                #print("\nGoal is in bottom-left, rotate 90° counter clockwise")
                # Swap and flip coordinates      
                new_x = grid_size - y
                new_y = x
                new_goal_x = grid_size - goal_y
                new_goal_y = goal_x
                # Adjust orientation (rotate 90° clockwise)
                new_orientation = (orientation + 1) % 4
                
            elif is_right and not is_bottom:  # Goal is in top-right, rotate 90° clockwise
                # Swap and flip coordinates    
                #print("\nGoal is in top-right, rotate 90° clockwise")
                new_x = y
                new_y = grid_size - x
                new_goal_x = goal_y
                new_goal_y = grid_size - goal_x
                # Adjust orientation (rotate 90° counter-clockwise)
                new_orientation = (orientation + 3) % 4
            else :
                #print("bottom-righ no change needed")
                new_x = x
                new_y = y
                new_goal_x = goal_x
                new_goal_y = goal_y
                new_orientation = orientation
 
            # Create the transformed agent state
            New_agent_state = np.concatenate(([new_x, new_y, new_orientation, status, new_goal_x, new_goal_y] ,agent_state[6:12]), axis = 0)
            #print(len(New_agent_state))
            #print("Transform state : ", New_agent_state)
            if len(agent_state) == 42 :
                #print("long agent2 : ", len(agent_state[12:22]))
                agent_state_2 = self._rotate_agent_other_state(agent_state[12:22], is_right, is_bottom, grid_size)
                agent_state_3 = self._rotate_agent_other_state(agent_state[22:32], is_right, is_bottom, grid_size)
                agent_state_4 = self._rotate_agent_other_state(agent_state[32:], is_right, is_bottom, grid_size)
                complete_new_state_agent = np.concatenate((New_agent_state, agent_state_2, agent_state_3, agent_state_4),  axis = 0)
                #print("final length state agent: ", len(complete_new_state_agent))
            if len(agent_state) == 22:
                agent_state_2 = self._rotate_agent_other_state(agent_state[12:22], is_right, is_bottom, grid_size)
                complete_new_state_agent =  np.concatenate((New_agent_state, agent_state_2),  axis = 0)
            if len(agent_state) == 32 : 
                agent_state_2 = self._rotate_agent_other_state(agent_state[12:22], is_right, is_bottom, grid_size)
                agent_state_3 = self._rotate_agent_other_state(agent_state[32:], is_right, is_bottom, grid_size)
                complete_new_state_agent = np.concatenate((New_agent_state, agent_state_2, agent_state_3),  axis = 0)
            complete_new_state[agent_idx] = complete_new_state_agent
        return complete_new_state

    def _rotate_agent_other_state(self, agent_state, is_goal_right, is_goal_bottom, grid_size):
        """
        Rotate state information for another agent within communication range
        
        Args:
            agent_state (list): State for another agent [x, y, orientation, status, lidar_data...]
            is_goal_right (bool): Whether the goal is to the right of the starting corner
            is_goal_bottom (bool): Whether the goal is to the bottom of the starting corner
            grid_size_x (int): Size of the grid in x dimension
            grid_size_y (int): Size of the grid in y dimension
            
        Returns:
            list: Transformed other agent state
        """
        if len(agent_state) != 10:
            print("agent state dim : ", agent_state)
            raise  ValueError 
        if agent_state[0] == - 1 : #agent not in range or dead
            return agent_state

        x, y = agent_state[0], agent_state[1]
        orientation = agent_state[2]
        status = agent_state[3]
        lidar_data = agent_state[4:]  # All LIDAR readings without goal info
        
        # Apply coordinate transformation based on which corner the goal is in
        if not is_goal_right and not is_goal_bottom:  # Goal is top-left, rotate 180°
            # Flip coordinates
            new_x = grid_size - x
            new_y = grid_size - y
            # Adjust orientation (rotate 180°)
            new_orientation = (orientation + 2) % 4
            
        elif not is_goal_right and is_goal_bottom:  # Goal is bottom-left, rotate 90° counter-clockwise
            # Swap and flip coordinates
            new_x = y
            new_y = grid_size - x
            # Adjust orientation (rotate 90° counter-clockwise)
            new_orientation = (orientation + 3) % 4
            
        elif is_goal_right and not is_goal_bottom:  # Goal is top-right, rotate 90° clockwise
            # Swap and flip coordinates
            new_x = grid_size - y
            new_y = x
            # Adjust orientation (rotate 90° clockwise)
            new_orientation = (orientation + 1) % 4
            
        else:  # Goal is bottom-right, no rotation needed
            new_x = x
            new_y = y
            new_orientation = orientation
        
        # Create the transformed other agent state
        
        transformed_other_state = np.concatenate(([new_x, new_y, new_orientation, status] ,lidar_data), axis = 0)
        #print("Subsequent agent len : " ,len( transformed_other_state))
        return transformed_other_state

    def update_epsilon(self) -> None:
        """Update exploration rate with adaptive decay."""
        # Decrease epsilon faster early in training, slower later
        if self.episode_count < 100:
            # More aggressive decay in early episodes
            self.epsilon *= self.epsilon_decay
        else:
            # Slower decay in later episodes
            self.epsilon *= (self.epsilon_decay * 1.01)  # 1% slower decay
        
        # Ensure we don't go below minimum epsilon
        self.epsilon = max(self.epsilon, self.epsilon_min)

    def train_step(self) -> None:
        """Perform one training step."""
        
        # Sample from replay buffer
        batch = self.buffer.sample(self.batch_size)
        
        # Get batch dimensions
        batch_size = batch.central_state.shape[0]
        
        # Reshape central state to 2D grid for CNN processing
        batch_central_state_matrix = batch.central_state.view(
            batch_size, 1, self.max_grid_size, self.max_grid_size
        )
        batch_next_central_state_matrix = batch.next_central_state.view(
            batch_size, 1, self.max_grid_size, self.max_grid_size
        )
        
        # Compute target Q values
        y = self._compute_target(
            batch.reward, 
            batch.next_state, 
            batch_next_central_state_matrix,
            batch.done
        )
        
        # Process experience sequences for each agent
        all_agent_q_values = []
        
        for agent_idx in range(self.num_agents):
            # Reset Q-network hidden state for this sequence
            self.q_net.reset_hidden(batch_size=batch_size, device=self.device)
            
            # Get agent-specific states
            agent_states = batch.state[:, agent_idx]
            
            # Get Q values for this agent's actions
            agent_q_values = self.q_net(agent_states)
            
            # Select only the Q-values for actions actually taken
            agent_actions = batch.action[:, agent_idx].unsqueeze(1)
            agent_q_values = agent_q_values.gather(1, agent_actions)
            
            all_agent_q_values.append(agent_q_values)
        
        # Stack all agent Q-values
        q_values = torch.cat(all_agent_q_values, dim=1)
        
        # Use the mixer to get joint Q value
        q_tot = self.mixer(q_values, batch_central_state_matrix, batch.state)
        
        # Compute loss and update networks
        self.optimizer.zero_grad()
        loss = F.huber_loss(q_tot, y, delta=1.0)
        loss_value = loss.item()
            # Add loss to history if attribute exists
        if hasattr(self, 'loss_history'):
            self.loss_history.append(loss_value)
        
        
        # # Print loss occasionally
        # if len(self.loss_history) % 100 == 0:
        #     print(f"Training loss: {loss_value:.4f} (update #{len(self.loss_history)})")
    
        # Backpropagation with gradient clipping
        loss.backward()
        self.track_gradients()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), 0.5)
        self.optimizer.step()
        
        # Soft update target networks
        soft_update(self.q_net, self.target_q_net, self.tau)
        soft_update(self.mixer, self.target_mixer, self.tau)

    def reset_networks(self):
        """Reinitialize network weights with proper scaling."""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Kaiming initialization for ReLU networks
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
            # Apply to both networks
        self.q_net.apply(init_weights)
        self.mixer.apply(init_weights)
        
        # Reset target networks too
        self.target_q_net = copy.deepcopy(self.q_net)
        self.target_mixer = copy.deepcopy(self.mixer)

    def track_weight_changes(self):
        """Track changes in network weights to verify learning is happening."""
        # If first call, initialize weight tracking
        if not hasattr(self, 'weight_snapshots'):
            self.weight_snapshots = {
                'q_net': {},
                'mixer': {}
            }
            self.last_weight_check = 0

        # Only check every N episodes to reduce overhead
        if self.episode_count - self.last_weight_check < 10:
            return
        
        self.last_weight_check = self.episode_count
        
        # Check Q-network weights
        for name, param in self.q_net.named_parameters():
            # Calculate statistics of the weights
            param_data = param.data.detach().cpu()
            current_avg = param_data.mean().item()
            current_std = param_data.std().item()
            
            # Compare with previous snapshot if exists
            if name in self.weight_snapshots['q_net']:
                prev_avg = self.weight_snapshots['q_net'][name]['avg']
                prev_std = self.weight_snapshots['q_net'][name]['std']
                avg_change = abs(current_avg - prev_avg)
                std_change = abs(current_std - prev_std)
                
                print(f"Episode {self.episode_count}, Q-Net '{name}': " 
                    f"Avg change: {avg_change:.6f}, Std change: {std_change:.6f}")
            
            # Update snapshot
            self.weight_snapshots['q_net'][name] = {
                'avg': current_avg, 
                'std': current_std
            }
        
        # Also check mixer network weights
        for name, param in self.mixer.named_parameters():
            param_data = param.data.detach().cpu()
            current_avg = param_data.mean().item()
            current_std = param_data.std().item()
            
            if name in self.weight_snapshots['mixer']:
                prev_avg = self.weight_snapshots['mixer'][name]['avg']
                prev_std = self.weight_snapshots['mixer'][name]['std']
                avg_change = abs(current_avg - prev_avg)
                std_change = abs(current_std - prev_std)
                
                print(f"Episode {self.episode_count}, Mixer '{name}': " 
                    f"Avg change: {avg_change:.6f}, Std change: {std_change:.6f}")
            
            self.weight_snapshots['mixer'][name] = {
                'avg': current_avg, 
                'std': current_std
            }

    def track_gradients(self):
        """Track gradient magnitudes to detect vanishing/exploding gradients."""
        q_net_grad_norm = 0
        mixer_grad_norm = 0
        
        # Calculate gradient norm for Q-network
        for param in self.q_net.parameters():
            if param.grad is not None:
                q_net_grad_norm += param.grad.norm(2).item() ** 2
        q_net_grad_norm = q_net_grad_norm ** 0.5
        
        # Calculate gradient norm for mixer
        for param in self.mixer.parameters():
            if param.grad is not None:
                mixer_grad_norm += param.grad.norm(2).item() ** 2
        mixer_grad_norm = mixer_grad_norm ** 0.5
        
        # # Log to check if gradients are reasonable
        # print(f"Gradient norms - Q-Net: {q_net_grad_norm:.4f}, Mixer: {mixer_grad_norm:.4f}")
        
        # Check for vanishing/exploding gradients
        if q_net_grad_norm < 1e-4:
            print("WARNING: Q-network may have vanishing gradients!")
        elif q_net_grad_norm > 10.0:
            print("WARNING: Q-network may have exploding gradients!")
            
        if mixer_grad_norm < 1e-4:
            print("WARNING: Mixer may have vanishing gradients!")
        elif mixer_grad_norm > 10.0:
            print("WARNING: Mixer may have exploding gradients!")    
    

    @torch.no_grad()
    def _compute_target(
        self,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        next_central_state_matrix: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        """Compute target Q values for QMIX."""
        # Compute global reward (sum of individual rewards)
        global_reward = torch.sum(reward, dim=1, keepdim=True)
        
        # Reset target network hidden states for batch processing
        batch_size = next_state.shape[0]
        self.target_q_net.reset_hidden(batch_size=batch_size, device=self.device)
        
        # Compute target Q values
        next_target_qs = torch.stack(
            [self.target_q_net(next_state[:, i]) for i in range(self.num_agents)],
            dim=1,
        )
        
        # Select best actions (argmax)
        next_target_best_action = next_target_qs.argmax(dim=2, keepdim=True)
        
        # Get Q values for best actions
        next_target_best_q = next_target_qs.gather(2, next_target_best_action).squeeze(2)
        
        # Mix individual Q values to get joint Q value
        target_q_tot = self.target_mixer(
            next_target_best_q, 
            next_central_state_matrix,
            next_state
        )
        
        # Apply discount factor
        done = done.unsqueeze(1)
        y = global_reward + self.gamma * (1 - done) * target_q_tot
        
        return y