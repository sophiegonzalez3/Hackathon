import numpy as np

from env import MazeEnv


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


def sample_env() -> MazeEnv:
    config = sample_config()

    env = MazeEnv(
        size=config.get("grid_size"),
        walls_proportion=config.get("walls_proportion"), 
        num_dynamic_obstacles=config.get("num_dynamic_obstacles"),
        num_agents=config.get("num_agents"),
        communication_range=config.get("communication_range"),
        max_lidar_dist_main=config.get(
            "max_lidar_dist_main"
        ),
        max_lidar_dist_second=config.get(
            "max_lidar_dist_second"
        ),
        max_episode_steps=config.get(
            "max_episode_steps"
        ),
        render_mode=config.get("render_mode", None),
        seed=config.get("seed", None),  # Seed for reproducibility
    )

    return env
