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
        num_rnn_layers: int,
    ) -> None:
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dim = hidden_size
        self.num_rnn_layers = num_rnn_layers

        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_rnn_layers,
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
            h0 = torch.zeros(self.num_rnn_layers, batch_size, self.hidden_dim).to(
                state.device
            )
            hidden_state = h0

        z2, hidden_state = self.gru(z1, hidden_state)
        action = self.output_layer(z2)
        return action, hidden_state


class IDRQNAgent:
    def __init__(self, num_agents: int) -> None:
        print("loading IDRQNAgent")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng()
        self.num_agents = num_agents

        if self.num_agents < 2:
            msg = "Expecting at least 2 agents"
            raise ValueError(msg)

        self.individual_state_size = 10 * num_agents + 2
        self.state_size = self.individual_state_size
        self.action_size = 7
        self.hidden_size = 256
        self.num_rnn_layers = 1
        self.lr = 1e-3
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.9995
        self.buffer_size = 10_000
        self.sequence_length = 20
        self.batch_size = 64
        self.target_update_freq = 20

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

        # Create separate memory buffers for each agent
        self.memory = ReplayMemory(self.buffer_size, self.sequence_length)

        # Hidden states for each agent
        self.hidden_states = [None for _ in range(num_agents)]
        self.steps_done = 0

    def get_action(self, state_list: list, evaluation: bool = False):
        if evaluation:
            epsilon = 0
        else:
            epsilon = self.epsilon
            self.update_epsilon()

        if self.rng.random() < epsilon:
            return self.rng.choice(self.action_size, size=self.num_agents).tolist()

        actions = []
        for agent_id, numpy_state in enumerate(state_list):
            state = torch.from_numpy(numpy_state).to(self.device).unsqueeze(0).unsqueeze(0)
            q_values, self.hidden_states[agent_id] = self.policy_net(state, self.hidden_states[agent_id])
            action = q_values.squeeze(0).squeeze(0).argmax().cpu().item()
            actions.append(action)
        return actions

    def new_episode(self) -> None:
        """Reset hidden states and memories for all agents at the beginning of an episode"""
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

        # For each agent, sample from its memory and update the shared policy
        total_loss = 0
        agents_trained = 0

        for agent_id in range(self.num_agents):
            batch = self.memory.sample(self.batch_size)
            if batch is not None:
                states, actions, rewards, next_states, dones = batch
                loss = self.train_step(
                    states[:, :, agent_id],
                    actions[:, :, agent_id],
                    rewards[:, :, agent_id],
                    next_states[:, :, agent_id],
                    dones,
                )
                total_loss += loss
                agents_trained += 1

        # Update target network based on overall steps
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.steps_done += 1

        if agents_trained > 0:
            return total_loss / agents_trained
        return 0

    def store_transition(
        self, states: list, actions: list, rewards: list, next_states: list, done: bool
    ) -> None:
        """Store transition in the memory buffer of the specific agent"""
        # Convert to tensors
        state = torch.from_numpy(np.array(states)).to(self.device)
        action = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward = torch.tensor(rewards, device=self.device)
        next_state = torch.from_numpy(np.array(next_states)).to(self.device)
        done = torch.tensor(done, dtype=torch.int8, device=self.device)

        # Store in agent's memory
        self.memory.append(state, action, reward, next_state, done)

    @torch.no_grad()
    def compute_target(self, rewards, next_states, dones):
        next_q_values, _ = self.target_net(next_states)
        next_q_values = next_q_values.max(2)[0]

        # Set V(s_{t+1}) = 0 for terminal states
        next_q_values = next_q_values * (1 - dones.float())

        # Compute target Q values
        return rewards.squeeze(-1) + (self.gamma * next_q_values)

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> float:
        """Perform one step of optimization on the shared policy network"""

        # Compute V(s_{t+1}) for all next states - target network
        target_q_values = self.compute_target(rewards, next_states, dones)

        # Compute Q(s_t, a) - policy network
        q_values, _ = self.policy_net(states)
        corresponding_q_values = q_values.gather(2, actions.unsqueeze(2)).squeeze(2)

        # Compute loss
        loss = F.smooth_l1_loss(corresponding_q_values, target_q_values)

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