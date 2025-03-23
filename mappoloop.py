from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from scenario import sample_env
from utils import extract_central_state


class AgentProtocol(Protocol):
    def next_episode(self) -> None: ...

    def get_action(
        self,
        agent_state: npt.NDArray[np.float32],  # (num_agents, state_size)
        evaluation: bool = False,
    ) -> list[int]: ...

    def update_policy(
        self,
        agent_state: npt.NDArray[np.float32],  # (num_agents, state_size)
        agent_action: list[int],  # (num_agents,)
        rewards: npt.NDArray[np.float32],  # (num_agents,)
        agent_next_state: npt.NDArray[np.float32],  # (num_agents, state_size)
        done: bool,
        central_state: npt.NDArray[np.float32],  # (central_state_size,)
        next_central_state: npt.NDArray[np.float32],  # (central_state_size,)
    ) -> dict[str, Any]: ...


def train(agent: AgentProtocol, max_episodes: int) -> list[float]:
    all_rewards = []

    try:
        for episode in range(max_episodes):
            env = sample_env()
            agent_state, _ = env.reset()
            agent.next_episode()

            done = False
            total_reward = 0

            while not done:
                central_state = extract_central_state(env)
                agent_actions = agent.get_action(agent_state)

                next_agent_state, rewards, terminated, truncated, info = env.step(
                    agent_actions
                )
                next_central_state = extract_central_state(env)
                done = terminated or truncated
                total_reward += np.sum(rewards)

                agent.update_policy(
                    agent_state,
                    agent_actions,
                    rewards,
                    next_agent_state,
                    done,
                    central_state,
                    next_central_state,
                )

                agent_state = next_agent_state

                if done:
                    all_rewards.append(total_reward)
                    current_steps = info["current_step"]
                    num_evacuated = len(info["evacuated_agents"])
                    num_deactivated = len(info["deactivated_agents"])
                    print(
                        ", ".join(
                            [
                                f"{episode=}",
                                f"{current_steps=}",
                                f"{total_reward=}",
                                f"{num_evacuated=}",
                                f"{num_deactivated=}",
                            ]
                        )
                    )
                    env.close()
    except KeyboardInterrupt:
        print("Simulation interrupted by the user")
        env.close()

    return all_rewards


if __name__ == "__main__":
    from agent import RandomAgent

    random_agent = RandomAgent(num_agents=4)
    train(random_agent, max_episodes=2)
