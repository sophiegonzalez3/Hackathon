import numpy as np

from env import MazeEnv

MAX_GRID_SIZE = 30
MAX_NUM_AGENTS = 4
MAX_NUM_OBSTACLES = 5
CENTRAL_STATE_SIZE = (
    MAX_GRID_SIZE * MAX_GRID_SIZE + 6 + MAX_NUM_AGENTS * 5 + MAX_NUM_OBSTACLES * 2
)


def extract_central_state(env: MazeEnv) -> np.ndarray:
    parameters = np.array(
        [
            env.grid_size,
            env.communication_range,
            env.max_lidar_dist_main,
            env.max_lidar_dist_second,
            env.num_dynamic_obstacles,
            env.walls_proportion,
        ]
    )
    positions = np.hstack(env.agent_positions)
    lidar_orientations = np.hstack(env.lidar_orientation)
    evacuated_agents = np.array(
        [int(agent_id in env.evacuated_agents) for agent_id in range(env.num_agents)]
    )
    deactivated_agents = np.array(
        [int(agent_id in env.evacuated_agents) for agent_id in range(env.num_agents)]
    )

    grid = env.grid
    full_grid = np.ones((MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=np.float32)
    full_grid[: env.grid_size, : env.grid_size] = grid
    full_grid = full_grid.flatten()

    obstacles = -np.ones(2 * MAX_NUM_OBSTACLES)
    for i, obstacle in enumerate(env.dynamic_obstacles):
        obstacles[2 * i] = obstacle[0]
        obstacles[2 * i + 1] = obstacle[1]

    return np.hstack(
        (
            parameters,
            positions,
            lidar_orientations,
            evacuated_agents,
            deactivated_agents,
            obstacles,
            full_grid,
        )
    )
