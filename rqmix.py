import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from bufferqmix import ReplayMemory
from env import MazeEnv
from models import DRQNNetwork, MixingNetwork
from stateprocessing import (
    central_state_dynamic,
    central_state_static,
    rotate_state,
    rotate_agent_other_state,
)


class QMIXAgent:
    def __init__(
        self,
        num_agents: int,
        device: torch.device,
        hidden_size: int = 64,
        num_rnn_layers: int = 1,
        qmix_embedding_size: int = 64,
        qmix_hypernet_size: int = 64,
        buffer_size: int = 1000,
        batch_sequence_length: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
        target_update_freq: int = 1,
        tau: float = 5e-3,
        gradient_clipping_value: float | None = None,
    ) -> None:
        self.device = device
        self.rng = np.random.default_rng()
        self.num_agents = num_agents

        if self.num_agents < 2:
            msg = "Expecting at least 2 agents"
            raise ValueError(msg)

        self.individual_state_size = 10 * num_agents + 2
        self.central_state_size = 12 * num_agents + 901
        self.state_size = self.individual_state_size
        self.action_size = 7
        self.hidden_size = hidden_size
        self.num_rnn_layers = num_rnn_layers
        self.qmix_embedding_size = qmix_embedding_size
        self.qmix_hypernet_size = qmix_hypernet_size

        self.buffer_size = buffer_size
        self.batch_sequence_length = batch_sequence_length
        self.batch_size = batch_size

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.tau = tau

        # Initialize policy network for each agent
        self.policy_net = DRQNNetwork(
            self.state_size, self.hidden_size, self.action_size, self.num_rnn_layers
        ).to(self.device)
        self.target_net = DRQNNetwork(
            self.state_size, self.hidden_size, self.action_size, self.num_rnn_layers
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Initialize mixing networks
        self.mixer = MixingNetwork(
            num_agents,
            self.central_state_size,
            self.qmix_embedding_size,
            self.qmix_hypernet_size,
        ).to(self.device)
        self.target_mixer = MixingNetwork(
            num_agents,
            self.central_state_size,
            self.qmix_embedding_size,
            self.qmix_hypernet_size,
        ).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_mixer.eval()

        # Optimizer for policy network and mixer
        self.optimizer = optim.AdamW(
            list(self.policy_net.parameters()) + list(self.mixer.parameters()),
            lr=self.lr
        )

        # Create a memory buffer for QMIX
        self.memory = ReplayMemory(
            self.buffer_size,
            sequence_length=self.batch_sequence_length,
        )

        self.gradient_clipping_value = gradient_clipping_value

        # Hidden states for each agent
        self.hidden_states = [None for _ in range(num_agents)]
        self.episode_counter = 0
        self.steps_done = 0

        print("Using QMIX agent with mixer network")
        print(self.policy_net)
        print(self.mixer)

    def preprocess_state(self, state_list: list) -> list:
        rotated_state_list = rotate_state(state_list)
        return np.vstack(rotated_state_list).astype(np.float32)
        # return state_list
    
    def get_action(self, state_list: list, evaluation: bool = False):
        if evaluation:
            epsilon = 0
        else:
            epsilon = self.epsilon
            self.update_epsilon()

        if self.rng.random() < epsilon:
            return self.rng.choice(self.action_size, size=self.num_agents).tolist()

        preprocessed_state_list = self.preprocess_state(state_list)
        with torch.no_grad():
            actions = []
            for agent_id, state in enumerate(preprocessed_state_list):
                state_tensor = (
                    torch
                    .from_numpy(state)
                    .to(device=self.device, dtype=torch.float32)
                    .view((1, 1, -1))  # batch / step / state_size
                )
                q_values, self.hidden_states[agent_id] = self.policy_net(
                    state_tensor, self.hidden_states[agent_id]
                )
                action = q_values.squeeze(0).squeeze(0).argmax().cpu().item()
                actions.append(action)

        return actions

    def new_episode(self) -> None:
        """Reset hidden states and memories at the beginning of an episode"""
        self.episode_counter += 1
        self.hidden_states = [None for _ in range(self.num_agents)]
        self.memory.new_episode()

    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def extract_central_state(self, states_list: list, env: MazeEnv) -> np.array:
        # The agent retrieves the following `state` at each step:
        # - its own position and orientation (x, y, o);
        # - its status (0: still running, 1: evacuated, 2: deactivated);
        # - the position of its goal (x, y);
        # - the LIDAR data in the 3 directions (main, right, left):
        #     - the distance to the nearest obstacle (or the maximum LIDAR range if no obstacle is present);
        #     - the type of obstacle detected (0: no obstacle, 1: wall or grid boundary, 2: dynamic obstacle, 3: another agent).
        # - for each agent within the communication range:
        #     - their position and orientation;
        #     - their status;
        #     - their LIDAR data.
        central_state = []
        for state in states_list:
            central_state.extend(state[:12])

        central_state.append(
            env.communication_range
        )

        shape = env.grid.shape
        max_grid = np.ones(shape=(30, 30), dtype=np.int8)  # 1 is wall
        max_grid[:shape[0], :shape[1]] = env.grid
        central_state.extend(max_grid.flatten())

        # central_state_size = 12 * num_agents + 1 + 900
        return np.array(central_state, dtype=np.float32)

    def update_policy(
        self,
        states_list: list,
        actions_list: list,
        rewards_list: list,
        next_states_list: list,
        done: bool,
        env: MazeEnv,
    ) -> None:
        preprocessed_states_list = self.preprocess_state(states_list)
        preprocessed_next_states_list = self.preprocess_state(next_states_list)
        # Create global state for QMIX by concatenating all agent states
        central_state = self.extract_central_state(
            preprocessed_states_list, env
        )
        next_central_state = self.extract_central_state(
            preprocessed_next_states_list, env
        )

        # Store transitions with global state
        self.store_transition(
            preprocessed_states_list,
            actions_list,
            rewards_list,
            preprocessed_next_states_list,
            done,
            central_state,
            next_central_state
        )

        # Sample from memory
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return 0  # Not enough samples yet

        states, actions, rewards, next_states, dones, central_states, next_central_states = batch

        # QMIX: calculate joint Q-values using mixer network
        loss = self.train_qmix_step(states, actions, rewards, next_states, dones,
                                   central_states, next_central_states)

        # Update target networks
        if self.steps_done % self.target_update_freq == 0:
            self.soft_update(self.policy_net, self.target_net)
            self.soft_update(self.mixer, self.target_mixer)

        self.steps_done += 1
        return loss

    def soft_update(self, model, target_model):
        for param, target_param in zip(
            model.parameters(), target_model.parameters(), strict=True
        ):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def store_transition(
        self, states: list, actions: list, rewards: list, next_states: list, done: bool,
        central_state: np.ndarray, next_central_state: np.ndarray
    ) -> None:
        """Store transition in the memory buffer with global state for QMIX"""
        # Convert to tensors
        state = torch.from_numpy(np.array(states)).to(self.device)
        action = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward = torch.tensor(rewards, device=self.device)
        next_state = torch.from_numpy(np.array(next_states)).to(self.device)
        central_state_tensor = torch.from_numpy(central_state).to(self.device)
        next_central_state_tensor = torch.from_numpy(next_central_state).to(self.device)
        done = torch.tensor(done, dtype=torch.int8, device=self.device)

        # Store in memory (extended to include global state)
        self.memory.append(
            state, action, reward, next_state, done, central_state_tensor, next_central_state_tensor
        )

    @torch.no_grad()
    def compute_qmix_target(self, rewards, next_states, next_central_states, dones):
        """Compute the QMIX target using mixing network"""
        batch_size = next_states.size(0)
        seq_len = next_states.size(1)

        # Initialize individual next Q-values for each agent
        next_q_values_list = []

        # Calculate individual Q-values for each agent
        for agent_id in range(self.num_agents):
            # Get next Q values from target network for this agent
            next_q_values, _ = self.target_net(next_states[:, :, agent_id])
            next_max_q = next_q_values.max(2)[0].unsqueeze(2)  # [batch_size, seq_len, 1]
            next_q_values_list.append(next_max_q)

        # Stack all agents' Q-values
        next_max_q_stack = torch.cat(next_q_values_list, dim=2)  # [batch_size, seq_len, num_agents]

        # Use the target mixer to get joint Q-value
        joint_next_q = self.target_mixer(next_max_q_stack, next_central_states)  # [batch_size, seq_len, 1]

        # Set future value to 0 for terminal states
        joint_next_q = joint_next_q * (1 - dones.float().unsqueeze(-1))

        # Sum rewards across agents for joint reward
        joint_rewards = rewards.sum(dim=2, keepdim=True)

        # Compute the target
        target_q = joint_rewards + (self.gamma * joint_next_q)

        return target_q

    def train_qmix_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        central_states: torch.Tensor,
        next_central_states: torch.Tensor,
    ) -> float:
        """Perform QMIX training step - optimize the joint action-value function"""

        # Compute joint target Q value using target mixer
        target_q = self.compute_qmix_target(
            rewards, next_states, next_central_states, dones
        )

        # Compute individual current Q values for each agent
        agent_q_values_list = []

        for agent_id in range(self.num_agents):
            # Get Q values from policy network for this agent
            q_values, _ = self.policy_net(states[:, :, agent_id])

            # Select Q values for the actions that were actually taken
            agent_q = q_values.gather(2, actions[:, :, agent_id].unsqueeze(2))  # [batch_size, seq_len, 1]
            agent_q_values_list.append(agent_q)

        # Stack all agents' Q-values
        agent_q_stack = torch.cat(agent_q_values_list, dim=2)  # [batch_size, seq_len, num_agents]

        # Use mixer to get joint Q-value
        joint_q = self.mixer(agent_q_stack, central_states)  # [batch_size, seq_len, 1]

        # Compute loss on the joint Q values
        loss = F.smooth_l1_loss(joint_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        if self.gradient_clipping_value is not None:
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(),
                self.gradient_clipping_value,
            )
            torch.nn.utils.clip_grad_norm_(
                self.mixer.parameters(),
                self.gradient_clipping_value,
            )

        self.optimizer.step()

        return loss.item()

    def save(self, filename):
        """Save model weights and agent state"""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "mixer": self.mixer.state_dict(),
                "target_mixer": self.target_mixer.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps_done": self.steps_done,
            },
            filename,
        )

    def load(self, filename):
        """Load model weights and agent state"""
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_mixer.load_state_dict(checkpoint["target_mixer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps_done = checkpoint["steps_done"]