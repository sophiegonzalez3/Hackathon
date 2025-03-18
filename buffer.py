from dataclasses import dataclass

import torch


@dataclass
class Batch:
    state: torch.Tensor
    central_state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    next_central_state: torch.Tensor
    done: torch.Tensor


class Buffer:
    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        state_size: int,
        central_state_size: int,
        device: torch.device,
    ) -> None:
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.state_size = state_size
        self.central_state_size = central_state_size
        self.device = device

        self.state_buffer = torch.empty(
            (buffer_size, num_agents, state_size), device=self.device
        )
        self.central_state_buffer = torch.empty(
            (buffer_size, central_state_size), device=self.device
        )
        self.action_buffer = torch.empty(
            (buffer_size, num_agents), dtype=torch.int64, device=self.device
        )
        self.reward_buffer = torch.empty((buffer_size, num_agents), device=self.device)
        self.next_state_buffer = torch.empty(
            (buffer_size, num_agents, state_size), device=self.device
        )
        self.next_central_state_buffer = torch.empty(
            (buffer_size, central_state_size), device=self.device
        )
        self.done_buffer = torch.empty(
            (buffer_size,), dtype=torch.int8, device=self.device
        )

        self.current_index = 0
        self.current_size = 0

    def append(
        self,
        state: torch.Tensor,
        central_state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        next_central_state: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        num_states = state.shape[0]
        if self.current_index + num_states <= self.buffer_size:
            indices = slice(self.current_index, self.current_index + num_states)
            self.state_buffer[indices] = state
            self.central_state_buffer[indices] = central_state
            self.action_buffer[indices] = action
            self.reward_buffer[indices] = reward
            self.next_state_buffer[indices] = next_state
            self.next_central_state_buffer[indices] = next_central_state
            self.done_buffer[indices] = done

            self.current_index = (self.current_index + num_states) % self.buffer_size
            self.current_size = min(self.current_size + num_states, self.buffer_size)
        else:
            # Recursive call to append to first fill the buffer, and then add the rest
            filling_size = self.buffer_size - self.current_index

            self.append(
                state[:filling_size],
                central_state[:filling_size],
                action[:filling_size],
                reward[:filling_size],
                next_state[:filling_size],
                next_central_state[:filling_size],
                done[:filling_size],
            )

            self.append(
                state[filling_size:],
                central_state[filling_size:],
                action[filling_size:],
                reward[filling_size:],
                next_state[filling_size:],
                next_central_state[filling_size:],
                done[filling_size:],
            )

    def sample(
        self,
        batch_size: int,
    ) -> Batch:
        indices = torch.randint(0, self.current_size, (batch_size,), device=self.device)
        return Batch(
            state=self.state_buffer[indices],
            central_state=self.central_state_buffer[indices],
            action=self.action_buffer[indices],
            reward=self.reward_buffer[indices],
            next_state=self.next_state_buffer[indices],
            next_central_state=self.next_central_state_buffer[indices],
            done=self.done_buffer[indices],
        )

    def __len__(self) -> int:
        """Returns the number of stored transitions."""
        return self.current_size