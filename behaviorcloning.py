import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from heuristic import AstarAgent
from models import DRQNetwork
from scenario import sample_env


def run_episode_for_behavior_cloning() -> tuple[list, list]:
    env = sample_env()
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
        done = terminated or truncated

    env.close()

    return state_list, action_list


def clone_step(policy_net: DRQNetwork, optimizer: optim.Optimizer, device: torch.device) -> float:
    state_list, action_list = run_episode_for_behavior_cloning()

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
