import torch


class Buffer:
    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        state_size: int,
        action_size: int,
        device: torch.device
    ) -> None:
        """
        Initializes the replay buffer.
        
        Args:
            buffer_size (int): Maximum number of transitions to store.
            num_agents (int): Number of agents.
            state_size (int): Dimension of the state space.
            action_size (int): Dimension of the action space.
            device (torch.device): Device to store tensors (CPU/GPU).
        """
        self.buffer_size = buffer_size
        self.device = device
        self.states = torch.empty((buffer_size, num_agents, state_size), device=torch.device)
        self.actions = torch.empty((buffer_size, num_agents), dtype=torch.int64, device=torch.device)
        self.rewards = torch.empty((buffer_size, num_agents), device=torch.device)
        self.next_states = torch.empty((buffer_size, num_agents, state_size), device=torch.device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.int8, device=torch.device)

        self.current_index = 0
        self.current_size = 0

    def append(state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor) -> None:
        """
        Stores a transition in the buffer.
        
        Args:
            state (torch.Tensor): Tensor of shape (num_agents, state_size)
            action (torch.Tensor): Tensor of shape (num_agents,)
            reward (torch.Tensor): Tensor of shape (num_agents,)
            next_state (torch.Tensor): Tensor of shape (num_agents, state_size)
            done (torch.Tensor): Tensor of shape (num_agents,)
        """
        self.states[self.current_index] = state
        self.actions[self.current_index] = action
        self.rewards[self.current_index] = reward
        self.next_states[self.current_index] = next_state
        self.dones[self.current_index] = done

        self.current_index = (self.current_index + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a random batch from the buffer.
        
        Args:
            batch_size (int): Number of samples to retrieve.
        
        Returns:
            tuple:
                - states (torch.Tensor): Tensor of shape (batch_size, num_agents, state_size)
                - actions (torch.Tensor): Tensor of shape (batch_size, num_agents)
                - rewards (torch.Tensor): Tensor of shape (batch_size, num_agents)
                - next_states (torch.Tensor): Tensor of shape (batch_size, num_agents, state_size)
                - dones (torch.Tensor): Tensor of shape (batch_size, num_agents)
        """
        indices = torch.randint(0, self.current_size, (batch_size,), device=self.device)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self) -> int:
        """Returns the number of stored transitions."""
        return self.current_size