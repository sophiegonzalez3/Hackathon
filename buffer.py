from dataclasses import dataclass
import numpy as np
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

class SequenceBuffer:
    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        state_size: int,
        central_state_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> None:
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.state_size = state_size
        self.central_state_size = central_state_size
        self.sequence_length = sequence_length
        self.device = device

        # Same buffers as before
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
        
        # Add episode tracking for sequence sampling
        self.episode_buffer = torch.empty(
            (buffer_size,), dtype=torch.int64, device=self.device
        )
        self.step_buffer = torch.empty(
            (buffer_size,), dtype=torch.int64, device=self.device
        )

        self.current_index = 0
        self.current_size = 0
        self.current_episode = 0
        self.current_step = 0

    def __len__(self) -> int:
        """Returns the number of stored transitions."""
        return self.current_size
        
    def start_new_episode(self):
        """Call this at the start of each episode."""
        self.current_episode += 1
        self.current_step = 0
        
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
        # Increment step within episode
        self.current_step += 1
        
        # Store data as before
        index = self.current_index
        
        self.state_buffer[index] = state
        self.central_state_buffer[index] = central_state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_state
        self.next_central_state_buffer[index] = next_central_state
        self.done_buffer[index] = done
        
        # Store episode and step information
        self.episode_buffer[index] = self.current_episode
        self.step_buffer[index] = self.current_step

        self.current_index = (self.current_index + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)
        
    def sample_sequences(self, batch_size: int) -> Batch:
        """Sample sequences of experiences from the buffer."""
        # Find episode starting points that have at least sequence_length steps
        valid_episodes = {}
        
        # Build dictionary of episode lengths
        for i in range(self.current_size):
            idx = (self.current_index - 1 - i) % self.buffer_size  # Start from the most recent
            ep = self.episode_buffer[idx].item()
            
            if ep not in valid_episodes:
                valid_episodes[ep] = []
            
            valid_episodes[ep].append(idx)
        
        # Filter to episodes with enough steps
        valid_starting_indices = []
        for ep, indices in valid_episodes.items():
            if len(indices) >= self.sequence_length:
                # Sort indices by step
                sorted_indices = sorted(indices, key=lambda idx: self.step_buffer[idx].item())
                
                # Get starting indices where we can extract a full sequence
                for i in range(len(sorted_indices) - self.sequence_length + 1):
                    valid_starting_indices.append(sorted_indices[i])
        
        if len(valid_starting_indices) < batch_size:
            # Fall back to random sampling if not enough sequences
            return self.sample(batch_size)
            
        # Randomly select starting indices
        selected_starts = torch.tensor(
            self.rng.choice(valid_starting_indices, batch_size, replace=True),
            device=self.device
        )
        
        # Construct sequences
        state_batch = []
        central_state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        next_central_state_batch = []
        done_batch = []
        
        for start_idx in selected_starts:
            ep = self.episode_buffer[start_idx].item()
            step = self.step_buffer[start_idx].item()
            
            # Get all indices for this episode
            ep_indices = valid_episodes[ep]
            
            # Sort by step
            sorted_indices = sorted(ep_indices, key=lambda idx: self.step_buffer[idx].item())
            
            # Find where our start_idx is in the sorted list
            start_pos = sorted_indices.index(start_idx.item())
            
            # Get sequence indices
            seq_indices = sorted_indices[start_pos:start_pos + self.sequence_length]
            
            # Extract sequence data
            state_batch.append(self.state_buffer[seq_indices])
            central_state_batch.append(self.central_state_buffer[seq_indices])
            action_batch.append(self.action_buffer[seq_indices])
            reward_batch.append(self.reward_buffer[seq_indices])
            next_state_batch.append(self.next_state_buffer[seq_indices])
            next_central_state_batch.append(self.next_central_state_buffer[seq_indices])
            done_batch.append(self.done_buffer[seq_indices])
            
        # Stack into tensors
        return Batch(
            state=torch.stack(state_batch),
            central_state=torch.stack(central_state_batch),
            action=torch.stack(action_batch),
            reward=torch.stack(reward_batch),
            next_state=torch.stack(next_state_batch),
            next_central_state=torch.stack(next_central_state_batch),
            done=torch.stack(done_batch),
        )
        
    def sample(self, batch_size: int) -> Batch:
        """Sample individual transitions (fallback method)."""
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
    
    def prioritized_sample(self, batch_size):
        """Sample transitions with priority based on TD-error."""
        # If we don't have TD errors yet, use uniform sampling
        if not hasattr(self, 'td_errors') or len(self.td_errors) < self.current_size:
            return self.sample(batch_size)
        
        # Convert TD errors to priorities with a small positive constant
        priorities = np.abs(self.td_errors[:self.current_size]) + 1e-6
        
        # Sample with probability proportional to priority
        probs = priorities / np.sum(priorities)
        indices = np.random.choice(self.current_size, size=batch_size, p=probs)
        indices = torch.tensor(indices, device=self.device)
        
        return Batch(
            state=self.state_buffer[indices],
            central_state=self.central_state_buffer[indices],
            action=self.action_buffer[indices],
            reward=self.reward_buffer[indices],
            next_state=self.next_state_buffer[indices],
            next_central_state=self.next_central_state_buffer[indices],
            done=self.done_buffer[indices],
        )