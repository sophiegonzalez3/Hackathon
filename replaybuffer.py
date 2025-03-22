from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, num_agents: int) -> None:
        self.capacity = capacity
        self.num_agents = num_agents
        self.buffer = deque(maxlen=capacity)
        self.current_trajectory = []
        self.rng = np.random.default_rng()

    def store_transition(self, states, actions, action_log_probs, rewards, next_states, dones, values, advantages=None):
        self.current_trajectory.append({
            'states': states,
            'actions': actions,
            'action_log_probs': action_log_probs,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'values': values,
            'advantages': advantages
        })

    def end_trajectory(self, compute_advantages=True, gamma=0.99, lambda_=0.95):
        if compute_advantages and len(self.current_trajectory) > 0:
            self._compute_advantages(gamma, lambda_)

        if len(self.current_trajectory) > 0:
            self.buffer.append(self.current_trajectory)

        self.current_trajectory = []

    def _compute_advantages(self, gamma, lambda_):
        trajectory_length = len(self.current_trajectory)

        # Extract values and rewards
        values = [transition['values'] for transition in self.current_trajectory]
        rewards = [transition['rewards'] for transition in self.current_trajectory]
        dones = [transition['dones'] for transition in self.current_trajectory]

        # Initialize advantages and returns
        advantages = np.zeros((trajectory_length, self.num_agents), dtype=np.float32)
        returns = np.zeros((trajectory_length, self.num_agents), dtype=np.float32)

        # Compute GAE advantages
        last_advantage = np.zeros(self.num_agents, dtype=np.float32)
        for t in reversed(range(trajectory_length)):
            if t == trajectory_length - 1:
                # For the last step, just use the reward
                delta = rewards[t] - values[t]
                advantages[t] = delta
            else:
                # For other steps, use the next value and advantage
                delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
                advantages[t] = delta + gamma * lambda_ * (1 - dones[t]) * last_advantage

            last_advantage = advantages[t]
            returns[t] = advantages[t] + values[t]

        # Update trajectory with computed advantages and returns
        for t in range(trajectory_length):
            self.current_trajectory[t]['advantages'] = advantages[t]
            self.current_trajectory[t]['returns'] = returns[t]

    def sample(self, batch_size=None):
        if batch_size is None or batch_size >= len(self.buffer):
            return list(self.buffer)
        else:
            return self.rng.choices(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
        self.current_trajectory = []
