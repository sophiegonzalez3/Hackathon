import torch
from torch import nn
from torch.distributions import Categorical


class ActorNetwork(nn.Module):
    def __init__(
        self,
        state_size: int = 42,
        action_size: int = 7,
        hidden_size: int = 128,
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 1,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers

        self.device = device

        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(rnn_hidden_size, action_size)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(
            self.device
        )

    def forward(
        self, state: torch.Tensor, hidden_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # state: (batch_size, seq_len, state_size)
        # hidden_state: (rnn_num_layers, batch_size, rnn_hidden_size)
        batch_size = state.size(0)
        features = self.feature_layer(state)  # (batch_size, seq_len, hidden_size)

        if hidden_state is None:
            hidden_state = self.init_hidden(
                batch_size
            )  # (rnn_num_layers, batch_size, rnn_hidden_size)

        x, hidden_state = self.rnn(features, hidden_state)
        # x: (batch_size, seq_len, rnn_hidden_size)
        # hidden_state: (rnn_num_layers, batch_size, rnn_hidden_size)

        logits = self.output_layer(x)  # (batch_size, seq_len, action_size)

        return logits, hidden_state

    def get_action(
        self,
        state: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # state: (batch_size, seq_len, state_size)
        # hidden_state: (rnn_num_layers, batch_size, rnn_hidden_size)
        action_logits, new_hidden = self(state, hidden_state)
        dist = Categorical(logits=action_logits)
        if deterministic:
            actions = action_logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        action_log_prob = dist.log_prob(actions)
        return actions, action_log_prob, new_hidden

    def evaluate_action(
        self,
        state: torch.Tensor,
        hidden_state: torch.Tensor | None,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # state: (batch_size, seq_len, state_size)
        # hidden_state: (rnn_num_layers, batch_size, rnn_hidden_size)
        # action: (batch_size, seq_len, 1)
        logits, new_hidden = self.forward(state, hidden_state)
        dist = Categorical(logits=logits)
        action_log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()
        return action_log_prob, dist_entropy, new_hidden


class CriticNetwork(nn.Module):
    def __init__(
        self,
        central_state_size: int,
        num_agents: int = 4,
        agent_state_size: int = 42,
        agent_action_size: int = 7,
        hidden_size: int = 128,
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 1,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.central_state_size = central_state_size
        self.num_agents = num_agents
        self.agent_state_size = agent_state_size
        self.agent_action_size = agent_action_size
        self.hidden_size = hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.device = device

        self.central_encoder = nn.Sequential(
            nn.Linear(central_state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.agent_encoder = nn.Sequential(
            nn.Linear(num_agents * (agent_state_size + 1), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.rnn = nn.GRU(
            input_size=2 * hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(rnn_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(
            self.device
        )

    def forward(
        self,
        central_state: torch.Tensor,
        agent_state: torch.Tensor,
        agent_action: torch.Tensor,
        hidden_state_p: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # central_state: (batch_size, seq_len, central_state_size)
        # agent_state: (batch_size, seq_len, num_agents, agent_state_size)
        # agent_action: (batch_size, seq_len, num_agents)
        # hidden_state: (rnn_num_layers, batch_size, rnn_hidden_size)

        batch_size = central_state.shape[0]
        seq_len = central_state.shape[1]

        central_features = self.central_encoder(
            central_state
        )  # (batch_size, seq_len, hidden_size)

        # Combine each agent's observation and action, then encode
        agent_inputs = torch.cat([agent_state, agent_action.unsqueeze(3)], dim=3).view(
            batch_size, seq_len, self.num_agents * (self.agent_state_size + 1)
        )  # (batch_size, seq_len, num_agents * (state_size + 1))
        agent_features = self.agent_encoder(
            agent_inputs
        )  # (batch_size, seq_len, hidden_size)

        features = torch.cat(
            [central_features, agent_features], dim=2
        )  # (batch_size, seq_len, 2*hidden_size)

        hidden_state = (
            self.init_hidden(batch_size) if hidden_state_p is None else hidden_state_p
        )  # (rnn_num_layers, batch_size, rnn_hidden_size)

        x, hidden_state = self.rnn(features, hidden_state)
        # x: (batch_size, seq_len, rnn_hidden_size)
        # hidden_state: (rnn_num_layers, batch_size, rnn_hidden_size)

        value = self.output_layer(x)
        # value: (batch_size, seq_len, 1)

        return value, hidden_state
