import numpy as np

class MyAgent:
    def __init__(self, num_agents: int):        
        # Random number generator for random actions pick
        self.rng = np.random.default_rng()
        self.num_agents = num_agents

    def get_action(self, state: list, evaluation: bool = False):
        # Choose random action
        actions = self.rng.integers(low=0, high=6, size=self.num_agents)
        return actions.tolist()

    def update_policy(self, actions: list, state: list, reward: float):
        # Do nothing
        pass


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import json

# Define hyperparameters
class QMIXConfig:
    def __init__(self):
        # Model parameters
        self.hidden_dim = 64
        self.mixing_embed_dim = 32
        self.gamma = 0.99
        self.lr = 3e-4
        self.batch_size = 32
        self.buffer_size = 5000
        self.target_update_interval = 200
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.999
        self.grad_norm_clip = 10
        
        # Get from environment
        self.n_agents = 4  # Will be set from env
        self.n_actions = 7  # Will be set from env
        self.state_shape = None  # Will be set from env observation space
        
        # Training settings
        self.train_epochs = 100
        self.steps_per_epoch = 500
        self.eval_interval = 10
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Agent Network - Each agent has its own network to compute Q-values based on observations
class AgentNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AgentNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Mixing Network - Combines individual Q-values while ensuring monotonicity
class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, mixing_embed_dim):
        super(MixingNetwork, self).__init__()
        
        # Hypernetwork for the first layer weights
        self.hyper_w1 = nn.Linear(state_dim, mixing_embed_dim * n_agents)
        # Hypernetwork for the second layer weights
        self.hyper_w2 = nn.Linear(state_dim, mixing_embed_dim)
        
        # Hypernetwork for biases
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b2 = nn.Linear(state_dim, 1)
        
    def forward(self, agent_qs, states):
        # Get batch size
        batch_size = agent_qs.size(0)
        
        # First layer weights and bias
        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        
        # Reshape weights to matrix form
        w1 = w1.view(batch_size, -1, self.hyper_w1.out_features // agent_qs.size(1))
        
        # Apply first layer
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1).squeeze(1) + b1)
        
        # Second layer weights and bias
        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)
        
        # Reshape weights
        w2 = w2.view(batch_size, -1, 1)
        
        # Final output - team Q-value
        q_tot = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2
        
        return q_tot

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
    def add(self, state, actions, reward, next_state, done):
        self.buffer.append((state, actions, reward, next_state, done))
        
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards).reshape(-1, 1)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones).reshape(-1, 1))
        )
        
    def __len__(self):
        return len(self.buffer)

# QMIX Agent
class QMIXAgent:
    def __init__(self, config, env):
        self.config = config
        self.env = env
        
        # Update configuration based on environment
        self.config.n_agents = env.num_agents
        self.config.n_actions = env.action_space.n
        self.config.state_shape = env.single_agent_state_size
        
        # Initialize networks
        self.agent_networks = []
        self.target_agent_networks = []
        
        for _ in range(self.config.n_agents):
            # Create online networks
            agent_net = AgentNetwork(self.config.state_shape, self.config.hidden_dim, self.config.n_actions)
            agent_net.to(self.config.device)
            self.agent_networks.append(agent_net)
            
            # Create target networks
            target_net = AgentNetwork(self.config.state_shape, self.config.hidden_dim, self.config.n_actions)
            target_net.to(self.config.device)
            target_net.load_state_dict(agent_net.state_dict())
            target_net.eval()
            self.target_agent_networks.append(target_net)
        
        # Calculate state dimension for mixing network
        # For this environment, we'll use the combined states of all agents as the global state
        mixing_input_dim = self.config.state_shape * self.config.n_agents
        
        # Initialize mixing networks
        self.mixer = MixingNetwork(self.config.n_agents, mixing_input_dim, self.config.mixing_embed_dim)
        self.mixer.to(self.config.device)
        
        self.target_mixer = MixingNetwork(self.config.n_agents, mixing_input_dim, self.config.mixing_embed_dim)
        self.target_mixer.to(self.config.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_mixer.eval()
        
        # Initialize optimizer
        params = list(self.mixer.parameters())
        for agent_net in self.agent_networks:
            params += list(agent_net.parameters())
        self.optimizer = optim.Adam(params, lr=self.config.lr)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.batch_size)
        
        # Initialize epsilon for exploration
        self.epsilon = self.config.epsilon_start
        
        # For logging
        self.training_rewards = []
        self.training_losses = []
        self.eval_rewards = []
        self.step_counter = 0
    
    def get_action(self, states, evaluation=False):
        """Select actions for each agent based on epsilon-greedy policy"""
        actions = []
        
        # Convert states to tensor
        states_tensor = torch.FloatTensor(states).to(self.config.device)
        
        for i, agent_net in enumerate(self.agent_networks):
            if not evaluation and random.random() < self.epsilon:
                # Exploration: random action
                actions.append(random.randint(0, self.config.n_actions - 1))
            else:
                # Exploitation: best action based on Q-values
                with torch.no_grad():
                    q_values = agent_net(states_tensor[i])
                    actions.append(q_values.argmax().item())
        
        return actions
    
    def update(self):
        """Update networks using a batch from replay buffer"""
        if len(self.replay_buffer) < self.config.batch_size:
            return 0.0  # Not enough samples yet
            
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        states = states.to(self.config.device)
        actions = actions.to(self.config.device)
        rewards = rewards.to(self.config.device)
        next_states = next_states.to(self.config.device)
        dones = dones.to(self.config.device)
        
        batch_size = states.shape[0]
        
        # Get Q-values for each agent for the chosen actions
        chosen_action_qvals = []
        for i, agent_net in enumerate(self.agent_networks):
            q_values = agent_net(states[:, i])
            chosen_action_qval = q_values.gather(1, actions[:, i].unsqueeze(1))
            chosen_action_qvals.append(chosen_action_qval)
        
        # Stack Q-values
        chosen_action_qvals = torch.cat(chosen_action_qvals, dim=1)
        
        # Get the global state by concatenating all agent states
        global_states = states.reshape(batch_size, -1)
        
        # Calculate team Q-value from individual Q-values using mixing network
        team_qval = self.mixer(chosen_action_qvals, global_states)
        
        # Target Q-values
        target_max_qvals = []
        for i, target_net in enumerate(self.target_agent_networks):
            target_q_values = target_net(next_states[:, i])
            target_max_qval = target_q_values.max(dim=1, keepdim=True)[0]
            target_max_qvals.append(target_max_qval)
        
        # Stack target Q-values
        target_max_qvals = torch.cat(target_max_qvals, dim=1)
        
        # Global next state
        global_next_states = next_states.reshape(batch_size, -1)
        
        # Calculate target team Q-value
        target_team_qval = self.target_mixer(target_max_qvals, global_next_states)
        
        # Calculate target (Bellman equation)
        targets = rewards + (1 - dones) * self.config.gamma * target_team_qval
        
        # Calculate loss
        loss = F.mse_loss(team_qval, targets.detach())
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), self.config.grad_norm_clip)
        for agent_net in self.agent_networks:
            torch.nn.utils.clip_grad_norm_(agent_net.parameters(), self.config.grad_norm_clip)
        
        # Update
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_networks(self):
        """Update target networks"""
        for i in range(self.config.n_agents):
            self.target_agent_networks[i].load_state_dict(self.agent_networks[i].state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
    
    def train(self):
        """Main training loop"""
        # Initialize logging
        best_eval_reward = float('-inf')
        
        for epoch in range(self.config.train_epochs):
            epoch_rewards = 0
            epoch_losses = []
            
            # Reset the environment
            states, _ = self.env.reset()
            
            for step in range(self.config.steps_per_epoch):
                self.step_counter += 1
                
                # Select actions
                actions = self.get_action(states)
                
                # Take a step in the environment
                next_states, rewards, done, _, info = self.env.step(actions)
                
                # Store experience in replay buffer
                self.replay_buffer.add(states, actions, rewards, next_states, done)
                
                # Update networks
                loss = self.update()
                if loss > 0:
                    epoch_losses.append(loss)
                
                # Update target networks if needed
                if self.step_counter % self.config.target_update_interval == 0:
                    self.update_target_networks()
                
                # Decay exploration rate
                self.decay_epsilon()
                
                # Track rewards
                epoch_rewards += rewards
                
                # Move to the next state
                states = next_states
                
                # Break if episode is done
                if done:
                    break
            
            # Log training metrics
            mean_loss = np.mean(epoch_losses) if epoch_losses else 0
            self.training_rewards.append(epoch_rewards)
            self.training_losses.append(mean_loss)
            
            print(f"Epoch {epoch + 1}/{self.config.train_epochs}: " 
                  f"Reward: {epoch_rewards:.2f}, "
                  f"Loss: {mean_loss:.4f}, "
                  f"Epsilon: {self.epsilon:.2f}")
            
            # Evaluate agent
            if (epoch + 1) % self.config.eval_interval == 0:
                eval_reward = self.evaluate()
                self.eval_rewards.append(eval_reward)
                
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.save_model("best_model.pt")
                
                print(f"Evaluation reward: {eval_reward:.2f}")
        
        # Plot training progress
        self.plot_training_progress()
    
    def evaluate(self, num_episodes=5):
        """Evaluate the agent's performance"""
        total_rewards = 0
        
        for _ in range(num_episodes):
            states, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                actions = self.get_action(states, evaluation=True)
                next_states, rewards, done, _, _ = self.env.step(actions)
                episode_reward += rewards
                states = next_states
            
            total_rewards += episode_reward
        
        return total_rewards / num_episodes
    
    def plot_training_progress(self):
        """Plot training progress"""
        plt.figure(figsize=(12, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(self.training_rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(3, 1, 2)
        plt.plot(self.training_losses)
        plt.title('Training Losses')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        plt.subplot(3, 1, 3)
        eval_episodes = [i * self.config.eval_interval for i in range(len(self.eval_rewards))]
        plt.plot(eval_episodes, self.eval_rewards)
        plt.title('Evaluation Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.show()
    
    def save_model(self, path):
        """Save model weights"""
        # Create a dictionary to store all network states
        model_dict = {
            'mixer': self.mixer.state_dict(),
            'agents': [agent.state_dict() for agent in self.agent_networks],
            'config': vars(self.config)
        }
        torch.save(model_dict, path)
    
    def load_model(self, path):
        """Load model weights"""
        model_dict = torch.load(path)
        
        # Load mixer network
        self.mixer.load_state_dict(model_dict['mixer'])
        self.target_mixer.load_state_dict(model_dict['mixer'])
        
        # Load agent networks
        for i, agent_state in enumerate(model_dict['agents']):
            self.agent_networks[i].load_state_dict(agent_state)
            self.target_agent_networks[i].load_state_dict(agent_state)

# Main function to run the training
def main():
    import json
    import argparse
    from simulation_config import simulation_config
    
    parser = argparse.ArgumentParser(description='Train QMIX agent for drone navigation')
    parser.add_argument('--config', type=str, default='config.json', help='Path to environment configuration file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--render', action='store_true', help='Enable environment rendering')
    args = parser.parse_args()
    
    # Configure environment and agent
    env, _, _ = simulation_config(args.config, new_agent=False)
    
    # Set up QMIX config
    config = QMIXConfig()
    config.train_epochs = args.epochs
    config.batch_size = args.batch_size
    
    # Set render mode if specified
    if args.render:
        env.render_mode = 'human'
    
    # Create QMIX agent
    agent = QMIXAgent(config, env)
    
    # Train agent
    agent.train()
    
    # Save final model
    agent.save_model("final_model.pt")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()