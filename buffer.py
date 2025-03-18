import torch


class Buffer:
    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        state_size: int,
        device: torch.device,
    ) -> None:
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.state_size = state_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, num_agents, state_size), device=self.device
        )
        self.actions = torch.empty(
            (buffer_size, num_agents), dtype=torch.int64, device=self.device
        )
        self.rewards = torch.empty((buffer_size, num_agents), device=self.device)
        self.next_states = torch.empty(
            (buffer_size, num_agents, state_size), device=self.device
        )
        self.dones = torch.empty((buffer_size,), dtype=torch.int8, device=self.device)

        self.current_index = 0
        self.current_size = 0

    def append(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        num_states = state.shape[0]
        if self.current_index + num_states <= self.buffer_size:
            indices = slice(self.current_index, self.current_index + num_states)
            self.states[indices] = state
            self.actions[indices] = action
            self.rewards[indices] = reward
            self.next_states[indices] = next_state
            self.dones[indices] = done

            self.current_index = (self.current_index + num_states) % self.buffer_size
            self.current_size = min(self.current_size + num_states, self.buffer_size)
        else:
            # Recursive call to append to first fill the buffer, and then add the rest
            filling_size = self.buffer_size - self.current_index

            self.append(
                state[:filling_size],
                action[:filling_size],
                reward[:filling_size],
                next_state[:filling_size],
                done[:filling_size],
            )

            self.append(
                state[filling_size:],
                action[filling_size:],
                reward[filling_size:],
                next_state[filling_size:],
                done[filling_size:],
            )

    def sample(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, self.current_size, (batch_size,), device=self.device)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        """Returns the number of stored transitions."""
        return self.current_size
