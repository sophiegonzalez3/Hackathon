from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import optim

from models import ActorNetwork, CriticNetwork
from replaymemory import Episode, EpisodeBuffer, ReplayMemory


class RandomAgent:
    def __init__(self, num_agents: int) -> None:
        self.num_agents = num_agents

    def next_episode(self) -> None:
        pass

    def get_action(
        self,
        agent_state: npt.NDArray[np.float32],  # (num_agents, state_size)
        evaluation: bool = False,
    ) -> list[int]:
        return np.random.randint(0, 7, size=self.num_agents).tolist()

    def update_policy(
        self,
        agent_state: npt.NDArray[np.float32],  # (num_agents, state_size)
        agent_action: list[int],  # (num_agents,)
        rewards: npt.NDArray[np.float32],  # (num_agents,)
        agent_next_state: npt.NDArray[np.float32],  # (num_agents, state_size)
        done: bool,
        central_state: npt.NDArray[np.float32],  # (central_state_size,)
        next_central_state: npt.NDArray[np.float32],  # (central_state_size,)
    ) -> dict[str, Any]:
        pass


class MappoAgent:
    def __init__(
        self,
        num_agents: int,
        actor: ActorNetwork,
        actor_optimizer: optim.Optimizer,
        critic: CriticNetwork,
        critic_optimizer: optim.Optimizer,
        gamma: float,
        clip_value: float,
        batch_size: int,
        buffer_size: int,
        sequence_length: int,
        device: torch.device | None,
    ) -> None:
        self.num_agents = num_agents
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.sequence_length = sequence_length
        self.replaymemory = ReplayMemory(buffer_size, sequence_length, device)
        self.clip_value = clip_value
        self.device = device

        self.actor_hidden_states: list[torch.Tensor | None] = [
            None for _ in range(self.num_agents)
        ]
        self.critic_hidden_state: torch.Tensor | None = None

        self.current_episode_buffer = EpisodeBuffer()

    def next_episode(self) -> None:
        self.actor_hidden_states = [None for _ in range(self.num_agents)]
        self.critic_hidden_state = None

        if not self.current_episode_buffer.is_empty():
            episode = Episode(self.current_episode_buffer, self.device)
            self.replaymemory.append(episode)
            self.current_episode_buffer.clear()

    def get_action(
        self,
        agent_state: npt.NDArray[np.float32],  # (num_agents, state_size)
        evaluation: bool = False,
    ) -> list[int]:
        # agent_state is (num_agents, state_size)
        actions = []
        for agent_id, state in enumerate(agent_state):
            state_tensor = (
                torch.from_numpy(state).to(self.device).view(1, 1, -1)
            )  # (batch_size, sequence_length, state_size)
            action_tensor, _, self.actor_hidden_states[agent_id] = (
                self.actor.get_action(
                    state_tensor,
                    self.actor_hidden_states[agent_id],
                    deterministic=evaluation,
                )
            )
            # action_tensor is (batch_size, sequence_length)
            action = action_tensor.squeeze().cpu().item()
            actions.append(action)
        return actions

    def update_policy(
        self,
        agent_state: npt.NDArray[np.float32],  # (num_agents, state_size)
        agent_action: list[int],  # (num_agents,)
        rewards: npt.NDArray[np.float32],  # (num_agents,)
        agent_next_state: npt.NDArray[np.float32],  # (num_agents, state_size)
        done: bool,
        central_state: npt.NDArray[np.float32],  # (central_state_size,)
        next_central_state: npt.NDArray[np.float32],  # (central_state_size,)
    ) -> dict[str, Any]:
        self.store_transition(
            agent_state,
            agent_action,
            rewards,
            agent_next_state,
            done,
            central_state,
            next_central_state,
        )

        if len(self.replaymemory) < 1:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        # Sample sequences from replay memory
        batch = self.replaymemory.sample_sequences(self.batch_size)

        # Compute returns and advantages
        returns, advantages = self._compute_returns_and_advantages(batch)

        # Update actor
        actor_loss, entropy = self._update_actor(batch, advantages)

        # Update critic
        critic_loss = self._update_critic(batch, returns)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
        }

    def store_transition(
        self,
        agent_state: npt.NDArray[np.float32],  # (num_agents, state_size)
        agent_action: list[int],  # (num_agents,)
        rewards: npt.NDArray[np.float32],  # (num_agents,)
        agent_next_state: npt.NDArray[np.float32],  # (num_agents, state_size)
        done: bool,
        central_state: npt.NDArray[np.float32],  # (central_state_size,)
        next_central_state: npt.NDArray[np.float32],  # (central_state_size,)
    ) -> None:
        self.current_episode_buffer.append(
            agent_state=agent_state,
            action=agent_action,
            central_state=central_state,
            reward=rewards,
            done=np.array([done] * self.num_agents),
            actor_hidden_state=np.array(
                [
                    hs.detach().cpu().numpy()
                    if hs is not None
                    else np.zeros(
                        (self.actor.rnn_num_layers, 1, self.actor.rnn_hidden_size)
                    )
                    for hs in self.actor_hidden_states
                ]
            ),
            critic_hidden_state=self.critic_hidden_state.detach().cpu().numpy()
            if self.critic_hidden_state is not None
            else np.zeros((self.critic.rnn_num_layers, 1, self.critic.rnn_hidden_size)),
        )

    def _compute_returns_and_advantages(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute returns and advantages for the sampled batch."""
        # Get batch components
        central_states = batch[
            "central_states"
        ]  # (batch_size, seq_len, central_state_size)
        agent_states = batch[
            "agent_states"
        ]  # (batch_size, seq_len, num_agents, agent_state_size)
        actions = batch["actions"]  # (batch_size, seq_len, num_agents)
        rewards = batch["rewards"]  # (batch_size, seq_len, num_agents)
        dones = batch["dones"]  # (batch_size, seq_len, num_agents)
        sequence_lengths = batch["sequence_lengths"]  # (batch_size)

        batch_size = central_states.shape[0]
        seq_len = central_states.shape[1]

        # Initialize critic hidden state
        critic_hidden = None
        if "critic_hidden_states" in batch:
            # (rnn_num_layers, batch_size, rnn_hidden_size)
            critic_hidden = batch["critic_hidden_states"]

        # Compute values for all states in the batch
        values = []
        for t in range(seq_len):
            value_t, critic_hidden = self.critic(
                central_states[:, t : t + 1],
                agent_states[:, t : t + 1],
                actions[:, t : t + 1],
                critic_hidden,
            )
            values.append(value_t)

        # Concatenate values over time dimension
        values = torch.cat(values, dim=1)  # (batch_size, seq_len, 1)

        # Create mask based on sequence lengths
        mask = torch.zeros((batch_size, seq_len), device=self.device)
        for i, length in enumerate(sequence_lengths):
            mask[i, :length] = 1.0

        # Compute returns and advantages
        returns = torch.zeros_like(values)
        advantages = torch.zeros_like(values)

        # For each sequence in the batch
        for b in range(batch_size):
            length = sequence_lengths[b].item()

            # Get the value estimate for the last state
            # If the sequence ends with a terminal state (done=1), use 0
            # Otherwise, bootstrap from the estimated value
            last_state_done = dones[b, length - 1, 0].item()

            if last_state_done:
                # If terminal state, initialize with 0
                R = torch.zeros(1, device=self.device)
            else:
                # If non-terminal, bootstrap from the value estimate
                # We use the last available value estimate
                R = values[b, length - 1].clone().detach()

            # Compute returns backward in time
            for t in reversed(range(length)):
                # Use mean reward across agents for global return calculation
                mean_reward = rewards[b, t].mean().unsqueeze(0)
                done = dones[b, t, 0].unsqueeze(0)

                # For the last step, we've already initialized R appropriately
                if t < length - 1:
                    R = mean_reward + self.gamma * R * (1 - done)
                else:
                    # For the last step, only use reward if terminal, otherwise already bootstrapped
                    if done.item() > 0:
                        R = mean_reward

                returns[b, t] = R

            # Compute advantages
            advantages[b, :length] = returns[b, :length] - values[b, :length]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def _update_actor(
        self, batch: dict[str, torch.Tensor], advantages: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the actor network."""
        # Get batch components
        agent_states = batch[
            "agent_states"
        ]  # (batch_size, seq_len, num_agents, agent_state_size)
        actions = batch["actions"]  # (batch_size, seq_len, num_agents)
        sequence_lengths = batch["sequence_lengths"]  # (batch_size)

        batch_size = agent_states.shape[0]
        seq_len = agent_states.shape[1]

        # Initialize zero loss and entropy
        policy_loss = 0
        entropy_loss = 0
        total_samples = 0

        # Update each agent's policy separately
        for agent_id in range(self.num_agents):
            # Get agent-specific data
            agent_states_i = agent_states[
                :, :, agent_id
            ]  # (batch_size, seq_len, agent_state_size)
            actions_i = actions[:, :, agent_id].long()  # (batch_size, seq_len)
            advantages_i = advantages[:, :, 0]  # (batch_size, seq_len)

            # Sample old action log probabilities
            actor_hidden = None
            if "actor_hidden_states" in batch:
                actor_hidden = batch["actor_hidden_states"][agent_id]

            # Get log probabilities and entropy
            old_log_probs = []
            for t in range(seq_len):
                log_prob_t, _, actor_hidden = self.actor.evaluate_action(
                    agent_states_i[:, t : t + 1],  # (batch_size, 1, agent_state_size)
                    actor_hidden,
                    actions_i[:, t : t + 1],  # (batch_size, 1)
                )
                old_log_probs.append(log_prob_t)

            # Concatenate log probs over time dimension
            old_log_probs = torch.cat(old_log_probs, dim=1)  # (batch_size, seq_len)

            # Detach old log probs to avoid computing gradients through them
            old_log_probs = old_log_probs.detach()

            # Compute new log probabilities and entropy
            actor_hidden = None
            if "actor_hidden_states" in batch:
                actor_hidden = batch["actor_hidden_states"][agent_id]

            log_probs = []
            entropy_total = 0
            for t in range(seq_len):
                log_prob_t, entropy_t, actor_hidden = self.actor.evaluate_action(
                    agent_states_i[:, t : t + 1],  # (batch_size, 1, agent_state_size)
                    actor_hidden,
                    actions_i[:, t : t + 1],  # (batch_size, 1)
                )
                log_probs.append(log_prob_t)
                entropy_total += entropy_t

            # Concatenate log probs over time dimension
            log_probs = torch.cat(log_probs, dim=1)  # (batch_size, seq_len)

            # Create mask for valid sequence parts
            mask = torch.zeros((batch_size, seq_len), device=self.device)
            for i, length in enumerate(sequence_lengths):
                mask[i, :length] = 1.0

            # Compute the ratio of new and old action probabilities
            ratio = torch.exp(log_probs - old_log_probs)

            # Compute surrogate objectives
            surr1 = ratio * advantages_i
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_value, 1.0 + self.clip_value)
                * advantages_i
            )

            # Calculate the actor loss using the PPO clip objective
            agent_loss = -torch.min(surr1, surr2) * mask

            # Sum over valid sequence parts
            valid_samples = mask.sum()
            if valid_samples > 0:
                policy_loss += agent_loss.sum() / valid_samples
                entropy_loss += entropy_total / valid_samples

            total_samples += 1

        # Average loss across agents
        if total_samples > 0:
            policy_loss /= total_samples
            entropy_loss /= total_samples

        # Update actor network
        self.actor_optimizer.zero_grad()
        total_loss = policy_loss - 0.01 * entropy_loss  # Entropy bonus
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        return policy_loss, entropy_loss

    def _update_critic(
        self, batch: dict[str, torch.Tensor], returns: torch.Tensor
    ) -> torch.Tensor:
        """Update the critic network."""
        # Get batch components
        central_states = batch[
            "central_states"
        ]  # (batch_size, seq_len, central_state_size)
        agent_states = batch[
            "agent_states"
        ]  # (batch_size, seq_len, num_agents, agent_state_size)
        actions = batch["actions"]  # (batch_size, seq_len, num_agents)
        sequence_lengths = batch["sequence_lengths"]  # (batch_size)

        batch_size = central_states.shape[0]
        seq_len = central_states.shape[1]

        # Initialize critic hidden state
        critic_hidden = None
        if "critic_hidden_states" in batch:
            critic_hidden = batch["critic_hidden_states"]

        # Compute predicted values
        values = []
        for t in range(seq_len):
            value_t, critic_hidden = self.critic(
                central_states[:, t : t + 1],  # (batch_size, 1, central_state_size)
                agent_states[
                    :, t : t + 1
                ],  # (batch_size, 1, num_agents, agent_state_size)
                actions[:, t : t + 1],  # (batch_size, 1, num_agents)
                critic_hidden,
            )
            values.append(value_t)

        # Concatenate values over time dimension
        values = torch.cat(values, dim=1)  # (batch_size, seq_len, 1)

        # Create mask for valid sequence parts
        mask = torch.zeros((batch_size, seq_len, 1), device=self.device)
        for i, length in enumerate(sequence_lengths):
            mask[i, :length] = 1.0

        # Calculate the number of valid entries
        num_valid = mask.sum()

        # Normalize returns - only consider valid parts of sequences
        if num_valid > 0:
            # Compute mean and std of returns where mask is 1
            masked_returns = returns * mask
            returns_mean = masked_returns.sum() / num_valid
            returns_var = (
                ((masked_returns - returns_mean) ** 2) * mask
            ).sum() / num_valid
            returns_std = torch.sqrt(
                returns_var + 1e-8
            )  # Add small epsilon for numerical stability

            # Normalize returns
            normalized_returns = (returns - returns_mean) / returns_std

            # Apply normalization for critic loss computation
            critic_loss = (
                F.mse_loss(
                    (values * mask), (normalized_returns * mask), reduction="sum"
                )
                / num_valid
            )
        else:
            critic_loss = torch.tensor(0.0, device=self.device)

        # Update critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        return critic_loss
