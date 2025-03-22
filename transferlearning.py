import numpy as np

from agent import MyAgent
from scenario import sample_env


def train(agent: MyAgent, num_episodes: int, print_every: int = 20) -> MyAgent:
    env = sample_env()

    # Metrics to follow the performance
    all_rewards = []
    total_reward = 0
    all_losses = []
    losses = []
    episode_count = 0

    # Initial reset of the environment
    state, info = env.reset()
    agent.new_episode()
    try:
        while episode_count < num_episodes:
            # Determine agents actions
            actions = agent.get_action(state)

            # Execution of a simulation step
            next_state, rewards, terminated, truncated, info = env.step(actions)
            total_reward += np.sum(rewards)

            # Update agent policy
            done = terminated or truncated
            loss = agent.update_policy(state, actions, rewards, next_state, done, env)

            all_losses.append(loss)
            losses.append(loss)
            state = next_state

            # Display of the step information
            print(f"\rEpisode {episode_count + 1}, Step {info['current_step']}, "
                  f"Reward: {total_reward:.2f}, "
                  f"Evacuated: {len(info['evacuated_agents'])}, "
                  f"Deactivated: {len(info['deactivated_agents'])}, "
                  f"MEAN TD LOSS: {np.array(losses).mean():.2e}"
            , end='')

            # If the episode is terminated
            if terminated or truncated:
                print("\r")
                episode_count += 1
                all_rewards.append(total_reward)
                total_reward = 0
                losses = []

                if episode_count < num_episodes:
                    env = sample_env()
                    state, info = env.reset()
                    agent.new_episode()

    except KeyboardInterrupt:
        print("\nSimulation interrupted by the user")

    finally:
        env.close()

    return agent, all_rewards, all_losses
