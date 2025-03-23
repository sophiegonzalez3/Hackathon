import random
from collections import deque
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import torch

NDArray: TypeAlias = npt.NDArray[np.float32]


class EpisodeBuffer:
    def __init__(self) -> None:
        self.central_states = []
        self.agent_states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.actor_hidden_states = []
        self.critic_hidden_states = []

    def append(
        self,
        central_state: NDArray,
        agent_state: NDArray,
        action: list[int],
        reward: NDArray,
        done: NDArray,
        actor_hidden_state: NDArray | None = None,
        critic_hidden_state: NDArray | None = None,
    ) -> None:
        """
        Add a single transition to the buffer.

        Args:
            central_state: Central state (central_state_size,)
            agent_state: Agent states (num_agents, agent_state_size)
            action: Actions (num_agents,)
            reward: Rewards (num_agents,)
            done: Done flags (num_agents,)
            actor_hidden_state: Actor hidden states (num_agents, rnn_num_layers, rnn_hidden_size) (optional)
            critic_hidden_state: Critic hidden state (rnn_num_layers, rnn_hidden_size) (optional)
        """
        action = np.asanyarray(action)
        self.central_states.append(central_state)
        self.agent_states.append(agent_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

        if actor_hidden_state is not None:
            self.actor_hidden_states.append(actor_hidden_state)

        if critic_hidden_state is not None:
            self.critic_hidden_states.append(critic_hidden_state)

    def clear(self) -> None:
        """Clear all stored transitions."""
        self.central_states.clear()
        self.agent_states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.actor_hidden_states.clear()
        self.critic_hidden_states.clear()

    def is_empty(self) -> bool:
        """Check if the buffer is empty."""
        return len(self.central_states) == 0

    def __len__(self) -> int:
        """Return the number of transitions in the buffer."""
        return len(self.central_states)


class Episode:
    """
    Container for a complete episode, converting the lists from EpisodeBuffer to tensors.
    """

    def __init__(
        self,
        buffer: EpisodeBuffer,
        device: torch.device,
    ) -> None:
        """
        Convert an EpisodeBuffer to an Episode with torch tensors.

        Args:
            buffer: The buffer containing the trajectory
            device: Device to store tensors on
        """
        self.device = device

        # Convert lists to numpy arrays, then to torch tensors
        self.central_states = torch.tensor(
            np.array(buffer.central_states), dtype=torch.float32, device=device
        )

        self.agent_states = torch.tensor(
            np.array(buffer.agent_states), dtype=torch.float32, device=device
        )

        self.actions = torch.tensor(
            np.array(buffer.actions), dtype=torch.float32, device=device
        )

        self.rewards = torch.tensor(
            np.array(buffer.rewards), dtype=torch.float32, device=device
        )

        self.dones = torch.tensor(
            np.array(buffer.dones), dtype=torch.float32, device=device
        )

        # Only convert if we have hidden states
        if buffer.actor_hidden_states:
            # (sequence_length, num_agents, rnn_num_layers, rnn_hidden_size)
            self.actor_hidden_states = torch.tensor(
                np.array(buffer.actor_hidden_states),
                dtype=torch.float32,
                device=device,
            )
        else:
            self.actor_hidden_states = None

        if buffer.critic_hidden_states:
            self.critic_hidden_states = torch.tensor(
                np.array(buffer.critic_hidden_states),
                dtype=torch.float32,
                device=device,
            )
        else:
            self.critic_hidden_states = None

        # Store the length of the episode
        self.length = len(buffer)

    def get_transition(self, idx: int) -> dict[str, torch.Tensor]:
        transition = {
            "central_state": self.central_states[idx],
            "agent_state": self.agent_states[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "done": self.dones[idx],
        }

        if self.actor_hidden_states is not None:
            transition["actor_hidden_state"] = self.actor_hidden_states[idx]

        if self.critic_hidden_states is not None:
            transition["critic_hidden_state"] = self.critic_hidden_states[idx]

        return transition

    def get_sequence(self, start_idx: int, length: int) -> dict[str, torch.Tensor]:
        # Make sure we don't go out of bounds
        end_idx = min(start_idx + length, self.length)
        actual_length = end_idx - start_idx

        sequence = {
            "central_states": self.central_states[start_idx:end_idx],
            "agent_states": self.agent_states[start_idx:end_idx],
            "actions": self.actions[start_idx:end_idx],
            "rewards": self.rewards[start_idx:end_idx],
            "dones": self.dones[start_idx:end_idx],
            "sequence_length": actual_length,
        }

        if self.actor_hidden_states is not None:
            sequence["actor_hidden_states"] = self.actor_hidden_states[
                start_idx:end_idx
            ]

        if self.critic_hidden_states is not None:
            sequence["critic_hidden_states"] = self.critic_hidden_states[
                start_idx:end_idx
            ]

        return sequence

    def __len__(self) -> int:
        return self.length


class ReplayMemory:
    def __init__(
        self,
        capacity: int,
        sequence_length,
        device: torch.device,
    ) -> None:
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.device = device
        self.episodes = deque(maxlen=capacity)

    def append(self, episode: Episode) -> None:
        self.episodes.append(episode)

    def sample_episodes(self, batch_size: int) -> list[Episode]:
        return random.sample(list(self.episodes), min(batch_size, len(self.episodes)))

    def sample_sequences(
        self, batch_size: int, sequence_length: int | None = None
    ) -> dict[str, torch.Tensor]:
        if not self.episodes:
            raise ValueError("Cannot sample from an empty replay memory")

        if sequence_length is None:
            sequence_length = self.sequence_length

        # Initialize batch collections
        batch = {
            "central_states": [],
            "agent_states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "sequence_lengths": [],
        }

        has_actor_hidden = False
        has_critic_hidden = False

        # Sample and process sequences
        for _ in range(batch_size):
            # Sample a random episode
            episode = random.choice(self.episodes)

            # Get a processed sequence (padded if needed)
            sequence = self._process_sequence(episode, sequence_length)

            # Add to batch
            batch["central_states"].append(sequence["central_states"])
            batch["agent_states"].append(sequence["agent_states"])
            batch["actions"].append(sequence["actions"])
            batch["rewards"].append(sequence["rewards"])
            batch["dones"].append(sequence["dones"])
            batch["sequence_lengths"].append(sequence["sequence_length"])

            # Track hidden states if they exist
            if (
                "actor_hidden_states" in sequence
                and sequence["actor_hidden_states"] is not None
            ):
                if "actor_hidden_states" not in batch:
                    batch["actor_hidden_states"] = []
                batch["actor_hidden_states"].append(sequence["actor_hidden_states"])
                has_actor_hidden = True

            if (
                "critic_hidden_states" in sequence
                and sequence["critic_hidden_states"] is not None
            ):
                if "critic_hidden_states" not in batch:
                    batch["critic_hidden_states"] = []
                batch["critic_hidden_states"].append(sequence["critic_hidden_states"])
                has_critic_hidden = True

        # Stack tensors across batch dimension
        result = {
            "central_states": torch.stack(batch["central_states"]),
            "agent_states": torch.stack(batch["agent_states"]),
            "actions": torch.stack(batch["actions"]),
            "rewards": torch.stack(batch["rewards"]),
            "dones": torch.stack(batch["dones"]),
            "sequence_lengths": torch.tensor(
                batch["sequence_lengths"], dtype=torch.long, device=self.device
            ),
        }

        # Only include hidden states if they were present
        if has_actor_hidden:
            # Originally shape: (batch_size, time_steps, num_agents, 1, rnn_hidden_size)
            actor_hidden = torch.stack(
                [h[0, :, :, 0] for h in batch["actor_hidden_states"]]
            )  # batch_size, num_agents, num_rnn_layers, rnn_hidden_size

            # (num_agents, num_rnn_layers, batch_size, rnn_hidden_size)
            result["actor_hidden_states"] = actor_hidden.permute(1, 2, 0, 3).contiguous()

        if has_critic_hidden:
            # Originally shape: (batch_size, time_steps, num_rnn_layers, 1, rnn_hidden_size)
            critic_hidden = torch.stack(
                [h[0, :, 0] for h in batch["critic_hidden_states"]]
            )  # batch_size, num_rnn_layers, rnn_hidden_size
            result["critic_hidden_states"] = critic_hidden.transpose(0, 1).contiguous()

        return result

    def _process_sequence(
        self, episode: Episode, sequence_length: int
    ) -> dict[str, torch.Tensor]:
        if episode.length <= sequence_length:
            sequence = episode.get_sequence(0, episode.length)
            return self._pad_sequence(sequence, sequence_length)
        else:
            max_start_idx = episode.length - sequence_length
            start_idx = random.randint(0, max_start_idx)
            return episode.get_sequence(start_idx, sequence_length)

    def _pad_sequence(
        self, sequence: dict[str, torch.Tensor], target_length: int
    ) -> dict[str, torch.Tensor]:
        # Get current sequence length
        current_length = sequence["central_states"].shape[0]

        # If no padding needed, return original sequence
        if current_length >= target_length:
            return sequence

        # Get shapes from the sequence
        central_state_size = sequence["central_states"].shape[-1]
        num_agents = sequence["agent_states"].shape[1]
        agent_state_size = sequence["agent_states"].shape[-1]

        # Calculate padding length
        padding_length = target_length - current_length

        # Create and apply padding for standard fields
        padded = {}

        # Central states: (seq_len, central_state_size)
        central_padding = torch.zeros(
            (padding_length, central_state_size),
            dtype=torch.float32,
            device=self.device,
        )
        padded["central_states"] = torch.cat(
            [sequence["central_states"], central_padding], dim=0
        )

        # Agent states: (seq_len, num_agents, agent_state_size)
        agent_padding = torch.zeros(
            (padding_length, num_agents, agent_state_size),
            dtype=torch.float32,
            device=self.device,
        )
        padded["agent_states"] = torch.cat(
            [sequence["agent_states"], agent_padding], dim=0
        )

        # Actions: (seq_len, num_agents)
        action_padding = torch.zeros(
            (padding_length, num_agents), dtype=torch.float32, device=self.device
        )
        padded["actions"] = torch.cat([sequence["actions"], action_padding], dim=0)

        # Rewards: (seq_len, num_agents)
        reward_padding = torch.zeros(
            (padding_length, num_agents), dtype=torch.float32, device=self.device
        )
        padded["rewards"] = torch.cat([sequence["rewards"], reward_padding], dim=0)

        # Done flags: (seq_len, num_agents) - set to 1 (True) for padding
        done_padding = torch.ones(
            (padding_length, num_agents), dtype=torch.float32, device=self.device
        )
        padded["dones"] = torch.cat([sequence["dones"], done_padding], dim=0)

        # Preserve sequence length
        padded["sequence_length"] = sequence["sequence_length"]

        mask = torch.ones((target_length,), dtype=torch.bool, device=self.device)
        mask[current_length:] = False
        padded["mask"] = mask

        # Handle hidden states if available
        for key in ["actor_hidden_states", "critic_hidden_states"]:
            if key in sequence and sequence[key] is not None:
                # Get shape information
                hs_shape = sequence[key].shape
                # Create padding tensor with appropriate shape
                padding = torch.zeros(
                    (padding_length,) + hs_shape[1:],
                    dtype=torch.float32,
                    device=self.device,
                )
                padded[key] = torch.cat([sequence[key], padding], dim=0)

        return padded

    def sample_transitions(self, batch_size: int) -> dict[str, torch.Tensor]:
        if not self.episodes:
            raise ValueError("Cannot sample from an empty replay memory")

        batch = {
            "central_states": [],
            "agent_states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "actor_hidden_states": [],
            "critic_hidden_states": [],
        }

        has_actor_hidden = False
        has_critic_hidden = False

        # Sample transitions
        for _ in range(batch_size):
            # Sample a random episode
            episode = random.choice(self.episodes)

            # Sample a random transition
            idx = random.randint(0, episode.length - 1)
            transition = episode.get_transition(idx)

            batch["central_states"].append(transition["central_state"])
            batch["agent_states"].append(transition["agent_state"])
            batch["actions"].append(transition["action"])
            batch["rewards"].append(transition["reward"])
            batch["dones"].append(transition["done"])

            if (
                "actor_hidden_state" in transition
                and transition["actor_hidden_state"] is not None
            ):
                has_actor_hidden = True
                batch["actor_hidden_states"].append(transition["actor_hidden_state"])

            if (
                "critic_hidden_state" in transition
                and transition["critic_hidden_state"] is not None
            ):
                has_critic_hidden = True
                batch["critic_hidden_states"].append(transition["critic_hidden_state"])

        # Stack tensors across batch dimension
        result = {
            "central_states": torch.stack(batch["central_states"]),
            "agent_states": torch.stack(batch["agent_states"]),
            "actions": torch.stack(batch["actions"]),
            "rewards": torch.stack(batch["rewards"]),
            "dones": torch.stack(batch["dones"]),
        }

        # Only include hidden states if they were present
        if has_actor_hidden:
            result["actor_hidden_states"] = torch.stack(batch["actor_hidden_states"])

        if has_critic_hidden:
            result["critic_hidden_states"] = torch.stack(batch["critic_hidden_states"])

        return result

    def clear(self) -> None:
        """Clear all episodes from memory."""
        self.episodes.clear()

    def __len__(self) -> int:
        """Return the number of episodes in memory."""
        return len(self.episodes)
