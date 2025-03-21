import torch
import torch.nn.functional as F
from torch import nn


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
    def __init__(self, num_agents, central_state_size, embedding_dim, hypernet_dim):
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
