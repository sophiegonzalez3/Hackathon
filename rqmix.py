import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from bufferqmix import ReplayMemory
from env import MazeEnv


class DRQNNetwork(nn.Module):
    def __init__(
        self,
        state_size: int,
        hidden_size: int,
        action_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dim = hidden_size
        self.num_layers = num_layers

        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, action_size)

    def forward(
        self, state: torch.Tensor, hidden_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = state.size(0)
        sequence_length = state.size(1)

        state = state.view(-1, state.size(-1))
        z1 = self.feature_layer(state)
        z1 = z1.view(batch_size, sequence_length, -1)

        if hidden_state is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(
                state.device
            )
            hidden_state = h0

        z2, hidden_state = self.gru(z1, hidden_state)
        action = self.output_layer(z2)
        return action, hidden_state


class MixingNetwork(nn.Module):
    def __init__(self, num_agents, central_state_size, embedding_dim=128, hypernet_dim=128):
        super().__init__()
        self.num_agents = num_agents
        self.embedding_dim = embedding_dim
        self.central_state_size = central_state_size

        # Hypernetwork that produces the weights for the first layer of the mixing network
        self.hyper_w1 = nn.Sequential(
            nn.Linear(central_state_size, hypernet_dim),
            nn.ELU(),
            nn.Linear(hypernet_dim, num_agents * embedding_dim)
        )

        # Hypernetwork that produces the weights for the second layer of the mixing network
        self.hyper_w2 = nn.Sequential(
            nn.Linear(central_state_size, hypernet_dim),
            nn.ELU(),
            nn.Linear(hypernet_dim, embedding_dim)
        )

        # Hypernetwork that produces the bias for the first layer of the mixing network
        self.hyper_b1 = nn.Linear(central_state_size, embedding_dim)

        # Hypernetwork that produces the bias for the second layer of the mixing network
        self.hyper_b2 = nn.Sequential(
            nn.Linear(central_state_size, hypernet_dim),
            nn.ELU(),
            nn.Linear(hypernet_dim, 1)
        )

    def forward(self, agent_q_values, central_states):
        """
        Args:
            agent_q_values: Individual agent Q-values [batch_size, seq_len, num_agents]
            states: Global state for conditioning [batch_size, seq_len, state_size]
        Returns:
            joint_q_values: Joint action-value [batch_size, seq_len, 1]
        """
        batch_size = agent_q_values.size(0)
        seq_len = agent_q_values.size(1)

        # Reshape states for processing
        central_states = central_states.reshape(-1, self.central_state_size)  # [batch_size * seq_len, state_size]

        # Get weights and bias from hypernetworks
        w1 = self.hyper_w1(central_states)  # [batch_size * seq_len, num_agents * embedding_dim]
        w1 = w1.view(-1, self.num_agents, self.embedding_dim)  # [batch_size * seq_len, num_agents, embedding_dim]
        b1 = self.hyper_b1(central_states)  # [batch_size * seq_len, embedding_dim]

        w2 = self.hyper_w2(central_states)  # [batch_size * seq_len, embedding_dim]
        w2 = w2.view(-1, self.embedding_dim, 1)  # [batch_size * seq_len, embedding_dim, 1]
        b2 = self.hyper_b2(central_states)  # [batch_size * seq_len, 1]

        # Reshape agent q values for processing
        agent_q_values = agent_q_values.reshape(-1, 1, self.num_agents)  # [batch_size * seq_len, 1, num_agents]

        # Apply the mixing network with non-linearities
        # First layer: ensure positive weights for monotonicity
        w1 = torch.abs(w1)
        hidden = F.relu(torch.bmm(agent_q_values, w1) + b1.unsqueeze(1))  # [batch_size * seq_len, 1, embedding_dim]

        # Second layer: ensure positive weights for monotonicity
        w2 = torch.abs(w2)
        joint_q = torch.bmm(hidden, w2) + b2.unsqueeze(1)  # [batch_size * seq_len, 1, 1]

        # Reshape back
        joint_q = joint_q.view(batch_size, seq_len, 1)  # [batch_size, seq_len, 1]

        return joint_q


class QMIXAgent:
    def __init__(self, num_agents: int) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng()
        self.num_agents = num_agents

        if self.num_agents < 2:
            msg = "Expecting at least 2 agents"
            raise ValueError(msg)

        self.individual_state_size = 10 * num_agents + 2
        self.central_state_size = 12 * num_agents + 901

        self.state_size = self.individual_state_size
        self.action_size = 7
        self.hidden_size = 128
        self.num_rnn_layers = 1

        self.lr = 1e-3
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.9995
        self.buffer_size = 1_000
        self.sequence_length = 20
        self.batch_size = 64
        self.target_update_freq = 1

        # Initialize policy network for each agent
        self.policy_net = DRQNNetwork(
            self.state_size, self.hidden_size, self.action_size, self.num_rnn_layers
        ).to(self.device)
        self.target_net = DRQNNetwork(
            self.state_size, self.hidden_size, self.action_size, self.num_rnn_layers
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Initialize mixing networks
        self.mixer = MixingNetwork(num_agents, self.central_state_size).to(self.device)
        self.target_mixer = MixingNetwork(num_agents, self.central_state_size).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_mixer.eval()

        # Optimizer for policy network and mixer
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.mixer.parameters()), 
            lr=self.lr
        )

        # Create a memory buffer for QMIX
        self.memory = ReplayMemory(self.buffer_size, self.sequence_length)

        # Hidden states for each agent
        self.hidden_states = [None for _ in range(num_agents)]
        self.steps_done = 0

        print("Using QMIX agent with mixer network")
        print(self.policy_net)
        print(self.mixer)

    def get_action(self, state_list: list, evaluation: bool = False):
        # Convert all states to batch for efficient processing
        states_batch = np.array(state_list)
        states_tensor = torch.from_numpy(states_batch).to(self.device).unsqueeze(1)

        if evaluation:
            epsilon = 0
        else:
            epsilon = self.epsilon
            self.update_epsilon()

        if self.rng.random() < epsilon:
            return self.rng.choice(self.action_size, size=self.num_agents).tolist()

        # Use the shared policy net for all agents
        with torch.no_grad():
            actions = []
            for agent_id in range(self.num_agents):
                q_values, self.hidden_states[agent_id] = self.policy_net(
                    states_tensor[agent_id:agent_id+1], self.hidden_states[agent_id]
                )
                action = q_values.squeeze(0).squeeze(0).argmax().cpu().item()
                actions.append(action)

        return actions

    def new_episode(self) -> None:
        """Reset hidden states and memories at the beginning of an episode"""
        self.hidden_states = [None for _ in range(self.num_agents)]
        self.memory.new_episode()

    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def extract_central_state(self, states_list: list, env: MazeEnv) -> np.array:
        # The agent retrieves the following `state` at each step:
        # - its own position and orientation (x, y, o);
        # - its status (0: still running, 1: evacuated, 2: deactivated);
        # - the position of its goal (x, y);
        # - the LIDAR data in the 3 directions (main, right, left):
        #     - the distance to the nearest obstacle (or the maximum LIDAR range if no obstacle is present);
        #     - the type of obstacle detected (0: no obstacle, 1: wall or grid boundary, 2: dynamic obstacle, 3: another agent).
        # - for each agent within the communication range:
        #     - their position and orientation;
        #     - their status;
        #     - their LIDAR data.
        central_state = []
        for state in states_list:
            central_state.extend(state[:12])
        
        central_state.append(
            env.communication_range
        )

        shape = env.grid.shape
        max_grid = np.ones(shape=(30, 30), dtype=np.int8)  # 1 is wall
        max_grid[:shape[0], :shape[1]] = env.grid
        central_state.extend(max_grid.flatten())

        # central_state_size = 12 * num_agents + 1 + 900
        return np.array(central_state, dtype=np.float32)

    def update_policy(
        self,
        states_list: list,
        actions_list: list,
        rewards_list: list,
        next_states_list: list,
        done: bool,
        env: MazeEnv,
    ) -> None:
        # Create global state for QMIX by concatenating all agent states
        central_state = self.extract_central_state(states_list, env)
        next_central_state = self.extract_central_state(next_states_list, env)

        # Store transitions with global state
        self.store_transition(
            states_list, actions_list, rewards_list, next_states_list,
            done, central_state, next_central_state
        )

        # Sample from memory
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return 0  # Not enough samples yet

        states, actions, rewards, next_states, dones, central_states, next_central_states = batch

        # QMIX: calculate joint Q-values using mixer network
        loss = self.train_qmix_step(states, actions, rewards, next_states, dones,
                                   central_states, next_central_states)

        # Update target networks
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.steps_done += 1
        return loss

    def store_transition(
        self, states: list, actions: list, rewards: list, next_states: list, done: bool,
        central_state: np.ndarray, next_central_state: np.ndarray
    ) -> None:
        """Store transition in the memory buffer with global state for QMIX"""
        # Convert to tensors
        state = torch.from_numpy(np.array(states)).to(self.device)
        action = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward = torch.tensor(rewards, device=self.device)
        next_state = torch.from_numpy(np.array(next_states)).to(self.device)
        central_state_tensor = torch.from_numpy(central_state).to(self.device)
        next_central_state_tensor = torch.from_numpy(next_central_state).to(self.device)
        done = torch.tensor(done, dtype=torch.int8, device=self.device)

        # Store in memory (extended to include global state)
        self.memory.append(
            state, action, reward, next_state, done, central_state_tensor, next_central_state_tensor
        )

    @torch.no_grad()
    def compute_qmix_target(self, rewards, next_states, next_central_states, dones):
        """Compute the QMIX target using mixing network"""
        batch_size = next_states.size(0)
        seq_len = next_states.size(1)

        # Initialize individual next Q-values for each agent
        next_q_values_list = []

        # Calculate individual Q-values for each agent
        for agent_id in range(self.num_agents):
            # Get next Q values from target network for this agent
            next_q_values, _ = self.target_net(next_states[:, :, agent_id])
            next_max_q = next_q_values.max(2)[0].unsqueeze(2)  # [batch_size, seq_len, 1]
            next_q_values_list.append(next_max_q)

        # Stack all agents' Q-values
        next_max_q_stack = torch.cat(next_q_values_list, dim=2)  # [batch_size, seq_len, num_agents]

        # Use the target mixer to get joint Q-value
        joint_next_q = self.target_mixer(next_max_q_stack, next_central_states)  # [batch_size, seq_len, 1]

        # Set future value to 0 for terminal states
        joint_next_q = joint_next_q * (1 - dones.float().unsqueeze(-1))

        # Sum rewards across agents for joint reward
        joint_rewards = rewards.sum(dim=2, keepdim=True)

        # Compute the target
        target_q = joint_rewards + (self.gamma * joint_next_q)

        return target_q

    def train_qmix_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        central_states: torch.Tensor,
        next_central_states: torch.Tensor,
    ) -> float:
        """Perform QMIX training step - optimize the joint action-value function"""

        # Compute joint target Q value using target mixer
        target_q = self.compute_qmix_target(rewards, next_states, next_central_states, dones)

        # Compute individual current Q values for each agent
        agent_q_values_list = []

        for agent_id in range(self.num_agents):
            # Get Q values from policy network for this agent
            q_values, _ = self.policy_net(states[:, :, agent_id])

            # Select Q values for the actions that were actually taken
            agent_q = q_values.gather(2, actions[:, :, agent_id].unsqueeze(2))  # [batch_size, seq_len, 1]
            agent_q_values_list.append(agent_q)

        # Stack all agents' Q-values
        agent_q_stack = torch.cat(agent_q_values_list, dim=2)  # [batch_size, seq_len, num_agents]

        # Use mixer to get joint Q-value
        joint_q = self.mixer(agent_q_stack, central_states)  # [batch_size, seq_len, 1]

        # Compute loss on the joint Q values
        loss = F.smooth_l1_loss(joint_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Apply gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        # torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), 10)

        self.optimizer.step()

        return loss.item()

    def save(self, filename):
        """Save model weights and agent state"""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "mixer": self.mixer.state_dict(),
                "target_mixer": self.target_mixer.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps_done": self.steps_done,
            },
            filename,
        )

    def load(self, filename):
        """Load model weights and agent state"""
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_mixer.load_state_dict(checkpoint["target_mixer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps_done = checkpoint["steps_done"]