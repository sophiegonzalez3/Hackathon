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


class MyAgent:
    def __init__(self, num_agents: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Random number generator for random actions pick
        self.rng = np.random.default_rng()
        self.num_agents = num_agents
        self.action_low = 0
        self.action_high = 6

        self.state_size = 10 * num_agents + 2
        self.action_size = self.action_high + 1

        self.buffer_size = 10_000
        self.batch_size = 128
        self.lr = 1e-3
        self.gamma = 0.99
        self.tau = 0.005
        self.epsilon = 1.0
        self.epsilon_min = 1e-2
        self.epsilon_decay = 0.995

        self.hidden_size = 64
        self.layer_sizes = [
            self.state_size,
            self.hidden_size,
            self.hidden_size,
            self.action_size,
        ]
        self.buffer = Buffer(
            self.buffer_size, self.num_agents, self.state_size, self.device
        )
        self.models = [
            make_dnn(self.layer_sizes).to(self.device) for _ in range(self.num_agents)
        ]
        self.target_models = [copy.deepcopy(model) for model in self.models]
        self.optimizers = [optim.AdamW(m.parameters(), self.lr) for m in self.models]

    def get_action(self, state: list, evaluation: bool = False):
        if evaluation:
            self.update_epsilon()

        actions = []
        for agent, raw_s in enumerate(state):
            if evaluation and self.rng.random() < self.epsilon:
                actions.append(self.rng.integers(self.action_low, self.action_high))
            else:
                s = torch.from_numpy(raw_s).to(self.device)
                a_scores = (
                    self.models[agent](s.unsqueeze(0)).squeeze(0).cpu().detach().numpy()
                )
                a = np.argmax(a_scores)
                actions.append(a)

        return actions

    def update_epsilon(self) -> None:
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

    def update_policy(
        self, actions: list, state: list, reward: float, next_state: list, done: list
    ):
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        state = state_to_tensor(state, self.device)
        reward = torch.tensor(reward, device=self.device)
        next_state = state_to_tensor(state, self.device)
        done = torch.tensor(done, dtype=torch.int8, device=self.device)
        self.buffer.append(state, actions, reward, next_state, done)

        if len(self.buffer) < self.batch_size:
            return

        for agent in range(self.num_agents):
            (
                batch_state,
                batch_action,
                batch_reward,
                batch_next_state,
                batch_done,
            ) = self.buffer.sample(self.batch_size)
            y = self.compute_target(agent, batch_reward, batch_next_state, batch_done)
            action = batch_action[:, agent].unsqueeze(1).to(torch.int64)
            qs = self.models[agent](batch_state[:, agent, :]).gather(1, action)

            self.optimizers[agent].zero_grad()
            loss = F.smooth_l1_loss(y, qs)
            loss.backward()
            self.optimizers[agent].step()

        for agent in range(self.num_agents):
            soft_update(self.models[agent], self.target_models[agent], self.tau)

    @torch.no_grad()
    def compute_target(
        self,
        agent: int,
        batch_reward: torch.Tensor,
        batch_next_state: torch.Tensor,
        batch_done: torch.Tensor,
    ) -> torch.Tensor:
        reward = batch_reward[:, agent].unsqueeze(1)
        next_state = batch_next_state[:, agent, :]
        done = batch_done.unsqueeze(1)
        next_target_action_scores = self.target_models[agent](next_state).reshape(
            (self.batch_size, -1)
        )
        next_qs = next_target_action_scores.max(1, keepdim=True).values
        return reward + (1 - done) * self.gamma * next_qs
