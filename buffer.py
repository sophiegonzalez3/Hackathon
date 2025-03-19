import random
from collections import deque
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Transition:
    state: torch.Tensor  # (num_agents, state_size)
    action: torch.Tensor  # (num_agents,)
    reward: torch.Tensor  # (num_agents,)
    next_state: torch.Tensor  # (num_agents, state_size)
    done: torch.Tensor  # ()


class EpisodeBuffer:
    def __init__(self) -> None:
        self.transitions: list[Transition] = []

    def append(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor
    ) -> None:
        self.transitions.append(Transition(state, action, reward, next_state, done))

    def __len__(self) -> int:
        return len(self.transitions)

    def is_empty(self) -> bool:
        return len(self.transitions) == 0


class ReplayMemory:
    def __init__(self, buffer_size: int, sequence_length: int) -> None:
        self.buffer_size = buffer_size
        self.sequence_length = sequence_length
        self.episodes = deque(maxlen=buffer_size)
        self.current_episode = EpisodeBuffer()

    def new_episode(self) -> None:
        if len(self.current_episode) > 0:
            self.episodes.append(self.current_episode)
        self.current_episode = EpisodeBuffer()

    def append(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        self.current_episode.append(state, action, reward, next_state, done)

    def sample(self, batch_size: int):
        if len(self.episodes) < batch_size:
            return None

        sampled_episodes = random.sample(self.episodes, batch_size)

        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []

        for episode in sampled_episodes:
            if len(episode.transitions) < self.sequence_length:
                continue

            start_idx = random.randint(
                0, len(episode.transitions) - self.sequence_length
            )
            transitions = episode.transitions[
                start_idx : start_idx + self.sequence_length
            ]

            states = torch.stack([t.state for t in transitions])
            actions = torch.stack([t.action for t in transitions])
            rewards = torch.stack([t.reward for t in transitions])
            next_states = torch.stack([t.next_state for t in transitions])
            dones = torch.stack([t.done for t in transitions])

            batch_states.append(states)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_next_states.append(next_states)
            batch_dones.append(dones)

        if not batch_states:
            return None

        return (
            torch.stack(batch_states),
            torch.stack(batch_actions),
            torch.stack(batch_rewards),
            torch.stack(batch_next_states),
            torch.stack(batch_dones),
        )

    def __len__(self):
        return len(self.episodes)
