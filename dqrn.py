import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from buffer import ReplayMemory


class DRQNNetwork(nn.Module):
    def __init__(
        self, state_size: int, hidden_size: int, action_size: int, num_layers: int = 1
    ) -> None:
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dim = hidden_size
        self.num_layers = num_layers

        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU()
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
        z = self.feature_layer(state)
        z = z.view(batch_size, sequence_length, -1)

        if hidden_state is None:
            h0 = torch.zeros(batch_size, self.num_layers, self.hidden_dim).to(
                state.device
            )
            hidden_state = h0

        z, hidden_state = self.gru(z, hidden_state)

        action = self.output_layer(z)
        return action, hidden_state


class DRQNAgent:
    def __init__(self, num_agents: int) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng()
        self.num_agents = num_agents
        self.state_size = 10 * num_agents + 2
        self.action_size = 7
        self.hidden_size = 128
        self.lr = 1e-3
        self.gamma = 0.99
        self.buffer_size = 10_000
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.9999
        self.sequence_length = 20
        self.batch_size = 1000
        self.target_update_freq = 1

        # Initialize Q networks
        self.policy_net = DRQNNetwork(
            self.state_size, self.hidden_size, self.action_size
        ).to(self.device)
        self.target_net = DRQNNetwork(
            self.state_size, self.hidden_size, self.action_size
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.memory = ReplayMemory(self.buffer_size, self.sequence_length)

        self.hidden_state = None
        self.steps_done = 0

    def get_action(self, state: list, evaluation: bool = False):
        if evaluation:
            epsilon = 0
        else:
            epsilon = self.epsilon
            self.update_epsilon()
        print(f"\t{epsilon=}")

        if self.rng.random() < epsilon:
            return np.random.randint(0, self.action_size - 1)

        state = torch.from_numpy(state[0]).to(self.device).unsqueeze(0).unsqueeze(0)

        q_values, self.hidden_state = self.policy_net(state, self.hidden_state)
        action = q_values.squeeze(0).squeeze(0).argmax().cpu().item()
        return action

    def new_episode(self) -> None:
        self.hidden_state = None
        self.memory.new_episode()

    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_policy(
        self, state: list, action: list, reward, next_state: list, done: bool
    ) -> None:
        self.store_transition(state, action, reward, next_state, done)

        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return

        self.train_step(self, batch)

    def store_transition(
        self, state: list, action: list, reward, next_state: list, done: bool
    ) -> None:
        """Store transition in replay buffer."""
        # Convert to tensors
        state_tensor = torch.FloatTensor(state)
        action_tensor = action
        reward_tensor = reward
        next_state_tensor = torch.FloatTensor(next_state)
        done_tensor = done

        # Store in memory
        self.memory.append(
            state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor
        )

    def train_step(self, batch):
        print("train_step")
        """Perform one step of optimization."""
        states, actions, rewards, next_states, dones = batch
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute Q(s_t, a) - policy network
        q_values, _ = self.policy_net(states)
        q_values = q_values.gather(2, actions.unsqueeze(2)).squeeze(2)

        # Compute V(s_{t+1}) for all next states - target network
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_states)
            next_q_values = next_q_values.max(2)[0]

            # Set V(s_{t+1}) = 0 for terminal states
            next_q_values = next_q_values * (1 - dones.float())

            # Compute target Q values
            target_q_values = rewards + (self.gamma * next_q_values)

        # Compute loss
        loss = F.smooth_l1_loss(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.steps_done += 1

        return loss.item()

    def save(self, filename):
        """Save model weights."""
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
        """Load model weights."""
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps_done = checkpoint["steps_done"]
