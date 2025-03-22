import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from env import MazeEnv
from models import ActorNetwork, CriticNetwork
from replaybuffer import ReplayBuffer


MAX_GRID_SIZE = 30
MAX_NUM_AGENTS = 4
MAX_NUM_OBSTACLES = 5
GLOBAL_STATE_SIZE = MAX_GRID_SIZE * MAX_GRID_SIZE + 6 + MAX_NUM_AGENTS * 5 + MAX_NUM_OBSTACLES * 2


def extract_global_state(env: MazeEnv) -> np.ndarray:
    parameters = np.array([
        env.grid_size,
        env.communication_range,
        env.max_lidar_dist_main,
        env.max_lidar_dist_second,
        env.num_dynamic_obstacles,
        env.walls_proportion,
    ])
    positions = np.hstack(env.agent_positions)
    lidar_orientations = np.hstack(env.lidar_orientation)
    evacuated_agents = np.array([int(agent_id in env.evacuated_agents) for agent_id in range(env.num_agents)])
    deactivated_agents = np.array([int(agent_id in env.evacuated_agents) for agent_id in range(env.num_agents)])
    
    grid = env.grid
    full_grid = np.ones((MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=np.float32)
    full_grid[:env.grid_size, :env.grid_size] = grid
    full_grid = full_grid.flatten()
    
    obstacles = -np.ones(2 * MAX_NUM_OBSTACLES)
    for i, obstacle in enumerate(env.dynamic_obstacles):
        obstacles[2*i] = obstacle[0]
        obstacles[2*i+1] = obstacle[1]
    
    return np.hstack((parameters, positions,lidar_orientations, evacuated_agents, deactivated_agents, obstacles, full_grid))


class MyAgent:
    def __init__(
        self,
        num_agents: int,
        device: torch.device,
        actor_net: ActorNetwork,
        actor_optim: optim.Optimizer,
        critic_net: CriticNetwork,
        critic_optim: optim.Optimizer,
    ):
        self.num_agents = num_agents
        self.state_size = state_size
        self.grid_size = grid_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.update_epochs = update_epochs

        # Set device
        self.device = device if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        print(f"Using device: {self.device}")

        self.actor_net = actor_net
        self.critic_net = critic_net
        
        # Set up optimizers
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size, num_agents)

        # Hidden states for actors
        self.hidden_states = [None for _ in range(num_agents)]

        # Training metrics
        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []

    def preprocess_state(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        return state

    def get_action(self, states, evaluation=False):
        with torch.no_grad():
            actions = []
            action_log_probs = []
            entropies = []

            # Process each agent's state
            for i, state in enumerate(states):
                # Skip deactivated or evacuated agents
                if np.all(state[:2] == -1):  # Check if agent is inactive
                    actions.append(0)  # Default action
                    action_log_probs.append(0)
                    entropies.append(0)
                    continue

                state_tensor = self.preprocess_state(state).unsqueeze(0)  # Add batch dimension

                action, action_log_prob, entropy, self.hidden_states[i] = self.actor.get_action(
                    state_tensor, 
                    self.hidden_states[i],
                    deterministic=evaluation
                )

                actions.append(action.item())
                action_log_probs.append(action_log_prob.item())
                entropies.append(entropy.item())

            return actions, action_log_probs, entropies

    def update_policy(self, states, actions, rewards, next_states, done, env):
        global_state = self.critic.extract_global_state(env)

        with torch.no_grad():
            value = self.critic(global_state).squeeze().numpy()

        self.memory.store_transition(
            states=states,
            actions=actions,
            action_log_probs=self.actor_log_probs if hasattr(self, 'actor_log_probs') else None,
            rewards=rewards,
            next_states=next_states,
            dones=done,
            values=value
        )

        if done:
            self.memory.end_trajectory(compute_advantages=True, gamma=self.gamma, lambda_=self.gae_lambda)

            if len(self.memory) >= self.batch_size:
                self._update_networks()
                return True

        self.actor_log_probs = self.current_log_probs if hasattr(self, 'current_log_probs') else None

        return False

    def _update_networks(self):
        """Update actor and critic networks using collected trajectories"""
        # Sample trajectories from buffer
        trajectories = self.memory.sample(self.batch_size)

        for _ in range(self.update_epochs):
            # Process each trajectory
            for trajectory in trajectories:
                self._update_from_trajectory(trajectory)

        # Clear memory after update
        self.memory.clear()

    def _update_from_trajectory(self, trajectory):
        """Update networks from a single trajectory"""
        # Extract data from trajectory
        states_batch = []
        actions_batch = []
        old_log_probs_batch = []
        returns_batch = []
        advantages_batch = []
        global_states_batch = []

        for transition in trajectory:
            states_batch.append(transition['states'])
            actions_batch.append(transition['actions'])
            if transition['action_log_probs'] is not None:
                old_log_probs_batch.append(transition['action_log_probs'])
            returns_batch.append(transition['returns'] if 'returns' in transition else transition['advantages'] + transition['values'])
            advantages_batch.append(transition['advantages'])

            # We need to recalculate global states since they weren't stored
            # This would typically be done with the environment, but for now we'll skip it
            # global_states_batch.append(self.critic.extract_global_state(env))

        # Convert to tensors
        states_batch = [torch.FloatTensor(np.array(states)).to(self.device) for states in zip(*states_batch)]
        actions_batch = torch.LongTensor(actions_batch).to(self.device)
        if old_log_probs_batch:
            old_log_probs_batch = torch.FloatTensor(old_log_probs_batch).to(self.device)
        returns_batch = torch.FloatTensor(returns_batch).to(self.device)
        advantages_batch = torch.FloatTensor(advantages_batch).to(self.device)

        # Normalize advantages
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
        # Update critic
        critic_loss = self._update_critic(returns_batch, global_states_batch)

        # Update actor
        actor_loss, entropy = self._update_actor(states_batch, actions_batch, old_log_probs_batch, advantages_batch)

        # Record metrics
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        self.entropies.append(entropy)

    def _update_actor(self, states, actions, old_log_probs, advantages):
        """Update actor network"""
        total_actor_loss = 0
        total_entropy = 0

        # Process each agent's data
        for agent_idx in range(self.num_agents):
            # Get current agent's data
            agent_states = states[agent_idx]
            agent_actions = actions[:, agent_idx]
            agent_old_log_probs = old_log_probs[:, agent_idx] if old_log_probs is not None else None
            agent_advantages = advantages[:, agent_idx]

            # Forward pass to get new action distribution
            action_probs, _ = self.actor(agent_states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(agent_actions)
            entropy = dist.entropy().mean()

            # Compute policy loss with clipping
            if agent_old_log_probs is not None:
                ratio = torch.exp(new_log_probs - agent_old_log_probs)
                surr1 = ratio * agent_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * agent_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
            else:
                # If we don't have old log probs, just use the new ones
                actor_loss = -(new_log_probs * agent_advantages).mean()

            # Add entropy bonus
            actor_loss = actor_loss - self.entropy_coef * entropy

            # Backprop and optimize
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer_actor.step()

            total_actor_loss += actor_loss.item()
            total_entropy += entropy.item()

        return total_actor_loss / self.num_agents, total_entropy / self.num_agents

    def _update_critic(self, returns, global_states):
        """Update critic network"""
        # Since we're using a centralized critic, we need global states
        # For now, let's assume we have them
        if not global_states:
            return 0.0

        total_value_loss = 0

        for returns_batch, global_state in zip(returns, global_states):
            # Forward pass to get value prediction
            values = self.critic(global_state)

            # Compute value loss
            value_loss = self.value_loss_coef * F.mse_loss(values, returns_batch.unsqueeze(1))

            # Backprop and optimize
            self.optimizer_critic.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer_critic.step()

            total_value_loss += value_loss.item()

        return total_value_loss / len(global_states) if global_states else 0.0

    def new_episode(self):
        """Reset agent state for a new episode"""
        # Reset hidden states
        self.hidden_states = [None for _ in range(self.num_agents)]

        # End current trajectory if any
        self.memory.end_trajectory()

        # Reset current log probs
        if hasattr(self, 'actor_log_probs'):
            del self.actor_log_probs
