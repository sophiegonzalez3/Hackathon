import copy
from itertools import pairwise

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn, optim

from buffer import Buffer


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
    for param, target_param in zip(
        model.parameters(), target_model.parameters(), strict=True
    ):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class QNet(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        self.layer_sizes = [state_size, hidden_size, hidden_size, action_size]
        self.dnn = make_dnn(self.layer_sizes)

    def forward(self, batch_state: torch.Tensor) -> torch.Tensor:
        return self.dnn(batch_state)


class MixingNetwork(nn.Module):
    """QMIX Mixing Network that enforces monotonicity constraint."""

    def __init__(
        self, num_agents: int, central_state_size: int, mixing_hidden_dim: int
    ) -> None:
        super().__init__()

        self.num_agents = num_agents
        self.state_size = central_state_size
        self.mixing_hidden_dim = mixing_hidden_dim

        # Hypernetworks generate the weights and biases for the mixing network
        # First layer weights (positive only) - num_agents x mixing_hidden_dim
        self.hyper_w1 = nn.Sequential(
            nn.Linear(central_state_size, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, num_agents * mixing_hidden_dim),
        )

        # First layer bias
        self.hyper_b1 = nn.Sequential(
            nn.Linear(central_state_size, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, mixing_hidden_dim),
        )

        # Second layer weights (positive only) - mixing_hidden_dim x 1
        self.hyper_w2 = nn.Sequential(
            nn.Linear(central_state_size, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, mixing_hidden_dim),
        )

        # Second layer bias
        self.hyper_b2 = nn.Sequential(
            nn.Linear(central_state_size, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_qs: Individual agent Q-values [batch_size, num_agents]
            states: Global state [batch_size, state_size]
        Returns:
            q_tot: Mixed Q-value [batch_size, 1]
        """
        batch_size = agent_qs.size(0)

        w1 = torch.abs(self.hyper_w1(states)).view(
            batch_size, self.num_agents, self.mixing_hidden_dim
        )
        b1 = self.hyper_b1(states).view(batch_size, 1, self.mixing_hidden_dim)

        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.view(batch_size, self.mixing_hidden_dim, 1)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)

        # Forward pass
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot.squeeze(2)


class MyAgent:
    def __init__(
        self,
        num_agents: int,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Random number generator for random actions pick
        self.rng = np.random.default_rng()
        self.num_agents = num_agents
        self.action_low = 0
        self.action_high = 6

        self.state_size = 10 * num_agents + 2
        self.central_state_size = num_agents * self.state_size
        self.action_size = self.action_high + 1
        self.q_net_hidden_size = 128
        self.mixing_hidden_size = 128

        self.buffer_size = 10_000
        self.batch_size = 256
        self.lr = 3e-4
        self.gamma = 0.999
        self.tau = 0.05
        self.epsilon = 1.0
        self.epsilon_min = 1e-2
        self.epsilon_decay = 0.9995

        self.buffer = Buffer(
            self.buffer_size,
            self.num_agents,
            self.state_size,
            self.central_state_size,
            self.device,
        )

        self.q_net = QNet(self.state_size, self.action_size, self.q_net_hidden_size).to(
            self.device
        )
        self.target_q_net = copy.deepcopy(self.q_net)
        self.target_q_net.eval()

        self.mixer = MixingNetwork(
            self.num_agents, self.central_state_size, self.mixing_hidden_size
        ).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.target_mixer.eval()

        self.optimizer = optim.Adam(
            list(self.q_net.parameters()) + list(self.mixer.parameters()), self.lr
        )

    def get_action(self, state: list, evaluation: bool = False):
        if not evaluation:
            self.update_epsilon()

        actions = []
        for agent, agent_state in enumerate(state):
            if (not evaluation) and self.rng.random() < self.epsilon:
                actions.append(self.rng.integers(self.action_low, self.action_high))
            else:
                agent_state_tensor = torch.from_numpy(agent_state).to(self.device)
                a_scores = (
                    self.q_net(agent_state_tensor.unsqueeze(0))
                    .squeeze(0)
                    .cpu()
                    .detach()
                    .numpy()
                )
                a = np.argmax(a_scores)
                actions.append(a)

        return actions

    def update_policy(
        self,
        actions: list,
        state: list,
        reward: list,
        next_state: list,
        done: bool,
    ):
        state_ = state_to_tensor(state, self.device)
        central_state = self.extract_central_state(state_)
        next_state_ = state_to_tensor(next_state, self.device)
        next_central_state = self.extract_central_state(next_state_)

        self.buffer.append(
            state=state_,
            central_state=central_state,
            action=torch.tensor(actions, device=self.device),
            reward=torch.tensor(reward, device=self.device),
            next_state=state_to_tensor(next_state, self.device),
            next_central_state=next_central_state,
            done=torch.tensor(int(done), device=self.device),
        )

        if len(self.buffer) < self.batch_size:
            return

        self.train_step()

    def update_epsilon(self) -> None:
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

    def extract_central_state(self, states: torch.Tensor) -> torch.Tensor:
        return states.flatten()

    @torch.no_grad()
    def compute_target(
        self,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        next_central_state: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        # (batch_size, 1)
        global_reward = torch.sum(reward, dim=1, keepdim=True)

        # (batch_size, num_agents, action_size)
        next_target_qs = torch.stack(
            [self.target_q_net(next_state[:, i]) for i in range(self.num_agents)],
            dim=1,
        )

        # (batch_size, num_agents, 1)
        next_target_best_action = next_target_qs.argmax(dim=2, keepdim=True)

        # (batch_size, num_agents)
        next_target_best_q = next_target_qs.gather(2, next_target_best_action).squeeze(
            2
        )

        target_q_tot = self.target_mixer(next_target_best_q, next_central_state)

        done = done.unsqueeze(1)

        # (batch_size, 1)
        y = global_reward + self.gamma * (1 - done) * target_q_tot
        return y

    def train_step(self) -> None:
        batch = self.buffer.sample(self.batch_size)

        y = self.compute_target(
            batch.reward, batch.next_state, batch.next_central_state, batch.done
        )

        # (batch_size, num_agents, action_size)
        qs = torch.stack(
            [self.q_net(batch.state[:, i]) for i in range(self.num_agents)], dim=1
        )

        # (batch_size, num_agents, 1)
        action = batch.action.unsqueeze(2)

        q_values = qs.gather(2, action).squeeze(2)

        # (batch_size, 1)
        q_tot = self.mixer(q_values, batch.central_state)

        self.optimizer.zero_grad()
        loss = F.smooth_l1_loss(q_tot, y)
        loss.backward()
        self.optimizer.step()

        soft_update(self.q_net, self.target_q_net, self.tau)
        soft_update(self.mixer, self.target_mixer, self.tau)
