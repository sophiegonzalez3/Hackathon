from itertools import pairwise

import numpy as np
import numpy.typing as npt
import torch
from torch import nn


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

        self.hidden_size = 32
        self.layer_sizes = [
            num_agents * self.state_size,
            self.hidden_size,
            self.hidden_size,
            num_agents * self.action_size,
        ]
        self.model = make_dnn(self.layer_sizes)

    def get_action(self, state: list, evaluation: bool = False):
        state = state_to_tensor(state, self.device)

        state = state.flatten()
        actions_score = self.model(state.unsqueeze(0)).squeeze(0)
        actions_score = actions_score.reshape((self.num_agents, self.action_size))
        actions = torch.argmax(actions_score, dim=1)

        return action_to_array(actions)

    def update_policy(self, actions: list, state: list, reward: float):
        # Do nothing
        pass
