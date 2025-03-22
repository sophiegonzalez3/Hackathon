import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


class ActorNetwork(nn.Module):
    def __init__(self, state_size, hidden_size, action_size, num_rnn_layers=1):
        super().__init__()
        self.input_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.rnn_layers = num_rnn_layers

        self.feature_extraction = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(
        self, x: torch.Tensor, hidden_state: torch.Tensor | None=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        seq_length = x.size(1)

        x = x.view(batch_size * seq_length, -1)
        features = self.feature_extraction(x)
        features = features.view(batch_size, seq_length, -1)

        if hidden_state is None:
            hidden_state = torch.zeros(self.rnn_layers, batch_size, self.hidden_size, device=x.device)

        output, hidden_state = self.gru(features, hidden_state)

        output = output.reshape(batch_size * seq_length, -1)
        action_logits = self.output_layer(output)
        action_logits = action_logits.view(batch_size, seq_length, -1)

        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs, hidden_state

    def get_action(
        self,
        state: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        action_probs, hidden_state = self.forward(state, hidden_state)

        dist = Categorical(action_probs)

        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            action = dist.sample()

        action_log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, action_log_prob, entropy, hidden_state


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, agent_obs_dim, agent_action_dim, hidden_dim=64):
        super().__init__()

        # Global state encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Agent encoder (processes each agent's observation and action)
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_obs_dim + agent_action_dim, hidden_dim),
            nn.ReLU()
        )

        # Multi-head attention for agent interactions
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, global_state, agent_obs, agent_actions):
        batch_size = global_state.shape[0]
        n_agents = agent_obs.shape[1]

        # Encode global state
        global_features = self.global_encoder(global_state)  # [batch_size, hidden_dim]

        # Combine each agent's observation and action, then encode
        agent_inputs = torch.cat([agent_obs, agent_actions], dim=2)  # [batch_size, n_agents, obs_dim+action_dim]
        agent_features = self.agent_encoder(agent_inputs)  # [batch_size, n_agents, hidden_dim]

        # Reshape for attention: [sequence_length, batch_size, hidden_dim]
        agent_features_t = agent_features.transpose(0, 1)

        # Apply self-attention across agents
        attn_output, _ = self.attention(
            agent_features_t, agent_features_t, agent_features_t
        )

        # Average agent representations after attention
        attn_output = attn_output.transpose(0, 1)  # [batch_size, n_agents, hidden_dim]
        agent_embedding = attn_output.mean(dim=1)  # [batch_size, hidden_dim]

        # Combine global and agent information
        combined = torch.cat([global_features, agent_embedding], dim=1)

        # Compute value
        value = self.value_head(combined)

        return value