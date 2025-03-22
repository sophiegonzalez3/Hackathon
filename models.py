import json
import os

import torch
import torch.nn.functional as F
from torch import nn


class DRQNetwork(nn.Module):
    def __init__(
        self,
        state_size: int,
        hidden_size: int,
        action_size: int,
        num_recurrent_layers: int,
    ) -> None:
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_recurrent_layers = num_recurrent_layers

        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_recurrent_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, action_size)

    def forward(
        self, state: torch.Tensor, hidden_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = state.size(0)
        sequence_length = state.size(1)

        z1 = self.feature_layer(state)
        z1 = z1.view(batch_size, sequence_length, -1)

        if hidden_state is None:
            h0 = torch.zeros(self.num_recurrent_layers, batch_size, self.hidden_size).to(
                state.device
            )
            hidden_state = h0

        z2, hidden_state = self.gru(z1, hidden_state)
        action = self.output_layer(z2)
        return action, hidden_state

    def save(self, path):
        architecture = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'hidden_size': self.hidden_size,
            'num_recurrent_layers': self.num_recurrent_layers,
        }

        with open(os.path.join(path, 'architecture.json'), 'w') as f:
            json.dump(architecture, f)

        weights_path = os.path.join(path, 'weights.pt')
        torch.save(self.state_dict(), weights_path)

        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        architecture_path = os.path.join(path, 'architecture.json')
        with open(architecture_path, 'r') as f:
            architecture = json.load(f)

        model = cls(
            state_size=architecture['state_size'],
            action_size=architecture['action_size'],
            hidden_size=architecture['hidden_size'],
            num_recurrent_layers=architecture['num_recurrent_layers'],
        )

        weights_path = os.path.join(path, 'weights.pt')
        model.load_state_dict(torch.load(weights_path, weights_only=True))

        model.eval()
        return model


class MixingNetwork(nn.Module):
    def __init__(self, num_agents, central_state_size, embedding_size, hypernet_size):
        super().__init__()
        self.num_agents = num_agents
        self.embedding_size = embedding_size
        self.hypernet_size = hypernet_size
        self.central_state_size = central_state_size

        # Hypernetwork that produces the weights for the first layer of the mixing network
        self.hyper_w1 = nn.Sequential(
            nn.Linear(central_state_size, hypernet_size),
            nn.ELU(),
            nn.Linear(hypernet_size, num_agents * embedding_size)
        )

        # Hypernetwork that produces the weights for the second layer of the mixing network
        self.hyper_w2 = nn.Sequential(
            nn.Linear(central_state_size, hypernet_size),
            nn.ELU(),
            nn.Linear(hypernet_size, embedding_size)
        )

        # Hypernetwork that produces the bias for the first layer of the mixing network
        self.hyper_b1 = nn.Linear(central_state_size, embedding_size)

        # Hypernetwork that produces the bias for the second layer of the mixing network
        self.hyper_b2 = nn.Sequential(
            nn.Linear(central_state_size, hypernet_size),
            nn.ELU(),
            nn.Linear(hypernet_size, 1)
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
        w1 = torch.abs(self.hyper_w1(central_states))  # [batch_size * seq_len, num_agents * embedding_size]
        b1 = self.hyper_b1(central_states)  # [batch_size * seq_len, embedding_size]

        w2 = torch.abs(self.hyper_w2(central_states))  # [batch_size * seq_len, embedding_size]
        b2 = self.hyper_b2(central_states)  # [batch_size * seq_len, 1]

        w1 = w1.view(-1, self.num_agents, self.embedding_size)
        w2 = w2.view(-1, self.embedding_size, 1)

        agent_q_values = agent_q_values.reshape(-1, 1, self.num_agents)  # [batch_size * seq_len, 1, num_agents]
        hidden = F.relu(torch.bmm(agent_q_values, w1) + b1.unsqueeze(1))  # [batch_size * seq_len, 1, embedding_size]

        joint_q = torch.bmm(hidden, w2) + b2.unsqueeze(1)  # [batch_size * seq_len, 1, 1]
        return joint_q.view(batch_size, seq_len, 1)  # [batch_size, seq_len, 1]

    def save(self, path):
        architecture = {
            'num_agents': self.num_agents,
            'central_state_size': self.central_state_size,
            'embedding_size': self.embedding_size,
            'hypernet_size': self.hypernet_size,
        }

        with open(os.path.join(path, 'architecture.json'), 'w') as f:
            json.dump(architecture, f)

        weights_path = os.path.join(path, 'weights.pt')
        torch.save(self.state_dict(), weights_path)

        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        architecture_path = os.path.join(path, 'architecture.json')
        with open(architecture_path, 'r') as f:
            architecture = json.load(f)

        model = cls(
            num_agents=architecture['num_agents'],
            central_state_size=architecture['central_state_size'],
            embedding_size=architecture['embedding_size'],
            hypernet_size=architecture['hypernet_size'],
        )

        weights_path = os.path.join(path, 'weights.pt')
        model.load_state_dict(torch.load(weights_path, weights_only=True))

        model.eval()
        return model