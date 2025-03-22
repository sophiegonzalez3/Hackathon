import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from env import MazeEnv
from heuristic import AstarAgent
from models import DRQNetwork


def run_episode_for_behavior_cloning(config: dict[str, int]) -> tuple[list, list]:
    env = MazeEnv(
        size=config.get("grid_size"),  # Grid size
        walls_proportion=config.get("walls_proportion"),  # Walls proportion in the grid
        num_dynamic_obstacles=config.get(
            "num_dynamic_obstacles"
        ),  # Number of dynamic obstacles
        num_agents=config.get("num_agents"),  # Number of agents
        communication_range=config.get(
            "communication_range"
        ),  # Maximum distance for agent communications
        max_lidar_dist_main=config.get(
            "max_lidar_dist_main"
        ),  # Maximum distance for main LIDAR scan
        max_lidar_dist_second=config.get(
            "max_lidar_dist_second"
        ),  # Maximum distance for secondary LIDAR scan
        max_episode_steps=config.get(
            "max_episode_steps"
        ),  # Number of steps before episode termination
        render_mode=config.get("render_mode", None),
        seed=config.get("seed", None),  # Seed for reproducibility
    )

    agent = AstarAgent()

    state_list = []
    action_list = []
    state, info = env.reset()
    done = False
    step = 0

    while not done:
        actions = agent.get_action(state)
        state_list.append(state)
        action_list.append(actions)
        state, _, terminated, truncated, _ = env.step(actions)
        step += 1
        done = terminated or truncated or (step >= config["max_episode_steps"])
    return state_list, action_list


def sample_config() -> dict[str, int | None]:
    rng = np.random.default_rng()
    return {
        "grid_size": rng.integers(10, 31),
        "walls_proportion": rng.uniform(0.1, 0.6),
        "num_dynamic_obstacles": rng.integers(0, 5),
        "num_agents": 4,
        "communication_range": rng.integers(5, 10),
        "max_lidar_dist_main": 5,
        "max_lidar_dist_second": rng.integers(1, 3),
        "max_episodes": 1,
        "max_episode_steps": 500,
        "render_mode": None
    }


def clone_step(policy_net: DRQNetwork, optimizer: optim.Optimizer, device: torch.device) -> float:
    config = sample_config()
    state_list, action_list = run_episode_for_behavior_cloning(config)

    state_array = np.stack(state_list)
    action_array = np.stack(action_list)

    states = torch.from_numpy(state_array).to(device).permute((1, 0, 2))
    actions = torch.from_numpy(action_array).to(device)

    q_values, _ = policy_net(states)

    # q_values is of shape: (num_agents, time_step, action_size)
    # actions is of shape: (time_step, num_agents)

    q_values = q_values.permute(1, 0, 2)  # (time_step, num_agents, action_size)
    q_values = q_values.reshape(-1, q_values.shape[2])  # (time_step x num_agents, action_size)
    logits = F.log_softmax(q_values, dim=1)

    actions = actions.flatten()

    optimizer.zero_grad()
    loss = F.cross_entropy(logits.reshape(-1, 7), actions.reshape(-1))
    loss.backward()
    optimizer.step()

    return loss.item()


def clone_behavior(
    policy_net: DRQNetwork,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_episodes: int,
    print_every: int = 20
) -> list[float]:
    losses = []
    try:
        for episode in range(num_episodes):
            loss = clone_step(policy_net, optimizer, device)
            losses.append(loss)
            if episode % print_every == 0:
                print(f"{episode=}, {loss=}")
    except KeyboardInterrupt:
        print("Interrupted")
    return losses
