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
        self.action_size = self.action_high + 1
        self.q_net_hidden_size = 64

        self.buffer_size = 10_000
        self.batch_size = 256
        self.lr = 1e-3
        self.gamma = 0.99
        self.tau = 0.005
        self.epsilon = 1.0
        self.epsilon_min = 1e-2
        self.epsilon_decay = 0.9995

        self.buffer = Buffer(
            self.buffer_size, self.num_agents, self.state_size, self.device
        )

        self.q_net = QNet(self.state_size, self.action_size, self.q_net_hidden_size).to(
            self.device
        )
        self.target_q_net = copy.deepcopy(self.q_net)
        self.target_q_net.eval()
        self.optimizer = optim.AdamW(self.q_net.parameters(), self.lr)

    def get_action(self, state: list, evaluation: bool = False):
        if not evaluation:
            self.update_epsilon()

        actions = []
        for agent, raw_s in enumerate(state):
            if (not evaluation) and self.rng.random() < self.epsilon:
                actions.append(self.rng.integers(self.action_low, self.action_high))
            else:
                s = torch.from_numpy(raw_s).to(self.device)
                a_scores = self.q_net(s.unsqueeze(0)).squeeze(0).cpu().detach().numpy()
                a = np.argmax(a_scores)
                actions.append(a)

        return actions

    def update_policy(
        self, actions: list, state: list, reward: list, next_state: list, done: bool
    ):
        self.store_transition(actions, state, reward, next_state, done)

        if len(self.buffer) < self.batch_size:
            return

        self.train_step()

    def store_transition(
        self, actions: list, state: list, reward: list, next_state: list, done: bool
    ) -> None:
        self.buffer.append(
            state=state_to_tensor(state, device=self.device),
            action=torch.tensor(actions, device=self.device),
            reward=torch.tensor(reward, device=self.device),
            next_state=state_to_tensor(next_state, self.device),
            done=torch.tensor(int(done), device=self.device),
        )

    def update_epsilon(self) -> None:
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

    @torch.no_grad()
    def compute_target(
        self, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        # (batch_size, 1)
        global_reward = torch.sum(rewards, dim=1, keepdim=True)

        # (batch_size, num_agents, action_size)
        next_target_qs = torch.stack(
            [self.target_q_net(next_states[:, i]) for i in range(self.num_agents)],
            dim=1,
        )

        # (batch_size, num_agents, 1)
        next_target_best_action = next_target_qs.argmax(dim=2, keepdim=True)

        # (batch_size, num_agents, 1)
        next_target_best_q = next_target_qs.gather(2, next_target_best_action)

        dones = dones.unsqueeze(1)

        # (batch_size, 1)
        y = global_reward + self.gamma * (1 - dones) * next_target_best_q.sum(dim=1)
        return y

    def train_step(self) -> None:
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        y = self.compute_target(rewards, next_states, dones)

        # (batch_size, num_agents, action_size)
        qs = torch.stack(
            [self.q_net(states[:, i]) for i in range(self.num_agents)], dim=1
        )

        # (batch_size, num_agents, 1)
        actions = actions.unsqueeze(2)

        # (batch_size, 1)
        q_values = qs.gather(2, actions).sum(dim=1)

        self.optimizer.zero_grad()
        loss = F.smooth_l1_loss(q_values, y)
        loss.backward()
        self.optimizer.step()

        soft_update(self.q_net, self.target_q_net, self.tau)
