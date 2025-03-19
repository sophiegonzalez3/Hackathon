import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from buffer import ReplayMemory


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


class VDNAgent:
    def __init__(self, num_agents: int) -> None:
        print("Loading VDN agent")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng()
        self.num_agents = num_agents

        if self.num_agents < 2:
            msg = "Expecting at least 2 agents"
            raise ValueError(msg)

        self.individual_state_size = 10 * num_agents + 2

        self.state_size = self.individual_state_size
        self.action_size = 7
        self.hidden_size = 128
        self.num_rnn_layers = 1

        self.lr = 1e-3
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.9995
        self.buffer_size = 10_000
        self.sequence_length = 20
        self.batch_size = 64
        self.target_update_freq = 1

        # Initialize shared Q network for all agents
        self.policy_net = DRQNNetwork(
            self.state_size, self.hidden_size, self.action_size, self.num_rnn_layers
        ).to(self.device)
        self.target_net = DRQNNetwork(
            self.state_size, self.hidden_size, self.action_size, self.num_rnn_layers
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Create a single memory buffer for VDN
        self.memory = ReplayMemory(self.buffer_size, self.sequence_length)

        # Hidden states for each agent
        self.hidden_states = [None for _ in range(num_agents)]
        self.steps_done = 0

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

    def update_policy(
        self,
        states_list: list,
        actions_list: list,
        rewards_list: list,
        next_states_list: list,
        done: bool
    ) -> None:
        self.store_transition(states_list, actions_list, rewards_list, next_states_list, done)

        # Sample from memory
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return 0  # Not enough samples yet
            
        states, actions, rewards, next_states, dones = batch
        
        # VDN: calculate individual Q-values for each agent
        loss = self.train_vdn_step(states, actions, rewards, next_states, dones)
        
        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.steps_done += 1
        return loss

    def store_transition(
        self, states: list, actions: list, rewards: list, next_states: list, done: bool
    ) -> None:
        """Store transition in the memory buffer"""
        # Convert to tensors
        state = torch.from_numpy(np.array(states)).to(self.device)
        action = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward = torch.tensor(rewards, device=self.device)
        next_state = torch.from_numpy(np.array(next_states)).to(self.device)
        done = torch.tensor(done, dtype=torch.int8, device=self.device)

        # Store in memory
        self.memory.append(state, action, reward, next_state, done)

    @torch.no_grad()
    def compute_vdn_target(self, rewards, next_states, dones):
        """Compute the VDN target: sum of individual target Q-values"""
        batch_size = next_states.size(0)
        seq_len = next_states.size(1)
        
        # Initialize total Q-values
        total_target_q_values = torch.zeros(batch_size, seq_len, device=self.device)
        
        # Calculate individual Q-values for each agent and sum them
        for agent_id in range(self.num_agents):
            # Get next Q values from target network for this agent
            next_q_values, _ = self.target_net(next_states[:, :, agent_id])
            next_max_q = next_q_values.max(2)[0]
            
            # Set future value to 0 for terminal states
            next_max_q = next_max_q * (1 - dones.float())
            
            # Add individual agent's contribution to total
            agent_target = rewards[:, :, agent_id].squeeze(-1) + (self.gamma * next_max_q)
            total_target_q_values += agent_target
            
        return total_target_q_values

    def train_vdn_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> float:
        """Perform VDN training step - optimize the joint action-value function"""
        
        # Compute joint target Q value (sum of individual targets)
        total_target_q = self.compute_vdn_target(rewards, next_states, dones)
        
        # Compute individual current Q values and sum them
        total_current_q = torch.zeros(states.size(0), states.size(1), device=self.device)
        
        for agent_id in range(self.num_agents):
            # Get Q values from policy network for this agent
            q_values, _ = self.policy_net(states[:, :, agent_id])
            
            # Select Q values for the actions that were actually taken
            agent_q = q_values.gather(2, actions[:, :, agent_id].unsqueeze(2)).squeeze(2)
            
            # Add to total Q value
            total_current_q += agent_q
        
        # Compute loss on the sum of Q values
        loss = F.smooth_l1_loss(total_current_q, total_target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def save(self, filename):
        """Save model weights and agent state"""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
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
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps_done = checkpoint["steps_done"]