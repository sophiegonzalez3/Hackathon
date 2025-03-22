import numpy as np
import torch


# def rotate_single_position(x: int, y:int, o: int|None, rotation):
#     if o is None:
#         return (new_x, new_y)

#     return (new_x, new_y, new_o)


# def rotate_state(state: list[int]) -> list[int]:
#     new_state = state.copy()
#     x, y, o = state[:3]
#     new_x, new_y, new_o = rotate_single_position(x, y, o)
#     new_gx, new_gy = rotate_single_position(gx, gy)

#     for i in range(12, len(state), 10):
#         x, y, o = state[i:i+3]


def central_state_static(
    # agent_states: torch.Tensor,
    grid_size: int,
    walls: set[tuple[int, int]],
    goal_area: list[tuple[int, int]],
    optimal_paths: list[tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    """
    Creates an enhanced central state for QMIX with separate matrices for static and dynamic elements.

    Args:
        agent_states: Individual agent states tensor [batch_size, num_agents, state_size]
        grid_size: Size of the environment grid
        walls: Set of wall coordinates (x, y)
        goal_area: List of goal area coordinates (x, y)
        optimal_paths: List of coordinates (x, y) representing optimal paths from A*
        device: Torch device

    Returns:
        Enhanced central state tensor [batch_size, 2, grid_size, grid_size]
    """

    # Fill the static environment matrix (channel 0)
    MAX_GRID_SIZE = 30
    needs_padding = grid_size < MAX_GRID_SIZE
    # print("\nGoal area : ", goal_area)

    static_matrix = torch.ones(
        (grid_size, grid_size), device=device
    )  # Default: +1 for neutral

    # Add walls: -5
    for wall in walls:

        static_matrix[wall] = -5

    # Add goals: +100
    for goal in goal_area:
        static_matrix[goal] = 100

    # Add optimal paths: +2
    for path_point in optimal_paths:
        # Only mark if not already a wall or goal
        if static_matrix[path_point] == 1:
            static_matrix[path_point] = 2

    # save_tensor_as_markdown(static_matrix, "staticBefore.md")
    # Detect the corner where the goal is currently located
    goal_center_x = goal_area[0][0]
    goal_center_y = goal_area[0][1]
    # print("goal center : " , goal_center_x, goal_center_y)

    # Determine which corner the goal is closest to
    is_bottom = goal_center_x >= grid_size / 2
    is_right = goal_center_y >= grid_size / 2

    # Rotate the matrix based on goal position to ensure goal is bottom right
    if not is_right and not is_bottom:  # Goal is in top-left, rotate 180°
        # print("Static Goal is in top-left, rotate 180°\n")
        static_matrix = torch.flip(static_matrix, [0, 1])
    elif (
        not is_right and is_bottom
    ):  # Goal is in bottom-left, rotate 90° counter clockwise
        static_matrix = torch.rot90(static_matrix, k=1, dims=[0, 1])
        # print("Static Goal is in bottom-left, rotate 90° counter-clockwise\n")
    elif is_right and not is_bottom:  # Goal is in top-right, rotate 90° clockwise
        static_matrix = torch.rot90(static_matrix, k=-1, dims=[0, 1])
        # print("Static Goal is in top-right, rotate 90° clockwise\n")
    else:
        pass
        # print("Static bottom-righ\n")
    # If goal is already bottom-right, no rotation needed

    if needs_padding:
        # Create a matrix with the target size, filled with wall values (-5)
        padded_matrix = torch.full((MAX_GRID_SIZE, MAX_GRID_SIZE), -5, device=device)

        # Place the actual grid in the top-left corner
        padded_matrix[:grid_size, :grid_size] = static_matrix

        # Use the padded matrix
        static_matrix = padded_matrix

    # Optional: save for visualization
    # save_tensor_as_markdown(static_matrix, "staticAfter.md")

    return static_matrix


def central_state_dynamic(
    agent_states: torch.Tensor,
    grid_size: int,
    agent_positions: list[np.ndarray],
    dynamic_obstacles: list[tuple[int, int]],
    evacuated_agents: set[int],
    deactivated_agents: set[int],
    comm_range: int,  # Added parameter for communication range
    goal_area: list[tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    """
    Creates the dynamic part of the enhanced central state for QMIX.
    Ensures the output tensor is always 30x30 by adding padding if the grid_size is smaller.

    Includes:
    - Halos around obstacles (-10 for adjacent cells)
    - Graduated positive halos around agents (from -5 at center to +5 at range extremity)

    Args:
        agent_states: Individual agent states tensor [batch_size, num_agents, state_size]
        grid_size: Size of the environment grid (can range from 10 to 30)
        agent_positions: List of agent positions as numpy arrays
        dynamic_obstacles: List of dynamic obstacle positions (x, y)
        evacuated_agents: Set of indices of evacuated agents
        deactivated_agents: Set of indices of deactivated agents
        comm_range: Communication range for agents (Manhattan distance)
        device: Torch device

    Returns:
        Enhanced dynamic central state tensor [grid_size, grid_size] or [30, 30] if padded
    """
    # Maximum grid size we want to support
    MAX_GRID_SIZE = 30

    # Check if we need padding
    needs_padding = grid_size < MAX_GRID_SIZE

    # Create the appropriate sized matrix for the actual grid
    dynamic_matrix = torch.ones(
        (grid_size, grid_size), device=device
    )  # Default: +1 for neutral

    # Add dynamic obstacles: -10
    for obstacle in dynamic_obstacles:
        x, y = obstacle
        dynamic_matrix[x, y] = -10

        # Add negative halo around obstacles
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                # Skip the center (obstacle itself)
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy
                # Check bounds
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    # Only overwrite if the cell is not already an obstacle
                    if dynamic_matrix[nx, ny] != -10:
                        dynamic_matrix[nx, ny] = -10

    # Add agents: -5 and graduated positive halos
    for i, pos in enumerate(agent_positions):
        if i not in evacuated_agents and i not in deactivated_agents:
            x, y = int(pos[0]), int(pos[1])

            # Mark agent position
            dynamic_matrix[x, y] = -5

            # Create graduated positive halo around agent
            for radius in range(1, comm_range + 1):
                # Calculate graduated value: -5 at center to +5 at extremity
                # Linear interpolation
                halo_value = -5 + (10 * radius / comm_range)

                # Iterate through all cells at Manhattan distance = radius
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        # Check if the point is exactly at Manhattan distance = radius
                        if abs(dx) + abs(dy) == radius:
                            nx, ny = x + dx, y + dy
                            # Check bounds
                            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                                # Prioritize: Keep obstacle values, but overwrite neutral values
                                if (
                                    dynamic_matrix[nx, ny] != -10
                                    and dynamic_matrix[nx, ny] != -5
                                ):
                                    dynamic_matrix[nx, ny] = halo_value

    # Detect the corner where the goal is currently located
    goal_center_x = goal_area[0][0]
    goal_center_y = goal_area[0][1]
    # print("goal center : " , goal_center_x, goal_center_y)

    # Determine which corner the goal is closest to
    is_bottom = goal_center_x >= grid_size / 2
    is_right = goal_center_y >= grid_size / 2

    # Rotate the matrix based on goal position to ensure goal is bottom right
    if not is_right and not is_bottom:  # Goal is in top-left, rotate 180°
        # print("\nGoal is in top-left, rotate 180°")
        dynamic_matrix = torch.flip(dynamic_matrix, [0, 1])
    elif (
        not is_right and is_bottom
    ):  # Goal is in bottom-left, rotate 90° counter clockwise
        dynamic_matrix = torch.rot90(dynamic_matrix, k=1, dims=[0, 1])
        # print("\nGoal is in bottom-left, rotate 90° counter clockwise")
    elif is_right and not is_bottom:  # Goal is in top-right, rotate 90° clockwise
        dynamic_matrix = torch.rot90(dynamic_matrix, k=-1, dims=[0, 1])
        # print("\nGoal is in top-right, rotate 90° clockwise")
    else:  # If goal is already bottom-right, on rotate pas
        # print("bottom-righ")
        pass

    if needs_padding:
        # Create a matrix with the target size, filled with wall values (-5)
        padded_matrix = torch.full((MAX_GRID_SIZE, MAX_GRID_SIZE), -5, device=device)

        # Place the actual grid in the top-left corner
        padded_matrix[:grid_size, :grid_size] = dynamic_matrix

        # Use the padded matrix
        dynamic_matrix = padded_matrix

    # Optional: save for visualization
    # save_tensor_as_markdown(dynamic_matrix, "dynamic.md")
    return dynamic_matrix


def rotate_agent_other_state(
    agent_state, is_goal_right, is_goal_bottom, grid_size
):
    """
    Rotate state information for another agent within communication range

    Args:
        agent_state (list): State for another agent [x, y, orientation, status, lidar_data...]
        is_goal_right (bool): Whether the goal is to the right of the starting corner
        is_goal_bottom (bool): Whether the goal is to the bottom of the starting corner
        grid_size_x (int): Size of the grid in x dimension
        grid_size_y (int): Size of the grid in y dimension

    Returns:
        list: Transformed other agent state
    """
    if len(agent_state) != 10:
        # print("agent state dim : ", agent_state)
        raise ValueError
    if agent_state[0] == -1:  # agent not in range or dead
        return agent_state

    x, y = agent_state[0], agent_state[1]
    orientation = agent_state[2]
    status = agent_state[3]
    lidar_data = agent_state[4:]  # All LIDAR readings without goal info

    # Apply coordinate transformation based on which corner the goal is in
    if not is_goal_right and not is_goal_bottom:  # Goal is top-left, rotate 180°
        # Flip coordinates
        new_x = grid_size - x
        new_y = grid_size - y
        # Adjust orientation (rotate 180°)
        new_orientation = (orientation + 2) % 4

    elif (
        not is_goal_right and is_goal_bottom
    ):  # Goal is bottom-left, rotate 90° counter-clockwise
        # Swap and flip coordinates
        new_x = grid_size - y
        new_y = x
        # Adjust orientation (rotate 90° clockwise)
        new_orientation = (orientation + 1) % 4
    
    elif (
        is_goal_right and not is_goal_bottom
    ):  # Goal is top-right, rotate 90° clockwise
        # Swap and flip coordinates
        new_x = y
        new_y = grid_size - x
        # Adjust orientation (rotate 90° counter-clockwise)
        new_orientation = (orientation + 3) % 4


    else:  # Goal is bottom-right, no rotation needed
        new_x = x
        new_y = y
        new_orientation = orientation

    # Create the transformed other agent state
    transformed_other_state = np.concatenate(
        ([new_x, new_y, new_orientation, status], lidar_data), axis=0
    )
    # print("Subsequent agent len : " ,len( transformed_other_state))
    return transformed_other_state


def rotate_state(state: list):
    num_agents = len(state)
    # Determine which corner the goal is closest to (only need first agent data for it)
    agent_0 = state[0]
    grid_size = np.max(
        state.flatten()
    )  ########## WARNING je suppose que c'est vrai et en plus que pour le goal pos est toujours la case la plus eloigne
    # print("State len for agent 0 : " , len(state[0]))
    # print("Deducted grid size : ", grid_size +1 )
    # print("initial state : " , state)
    goal_pos = agent_0[4:6]
    goal_center_x = goal_pos[0]
    goal_center_y = goal_pos[1]
    is_bottom = goal_center_x >= (grid_size + 1) / 2
    is_right = goal_center_y >= (grid_size + 1) / 2
    complete_new_state = np.empty(4, dtype=object)
    for agent_idx in range(num_agents):
        agent_state = state[agent_idx]
        if agent_state[0] == -1 or agent_state[1] == -1:  # Agent mort
            # print("agent mort")
            complete_new_state[agent_idx] = agent_state
            continue

        # Extract components from the agent state
        x, y = agent_state[0], agent_state[1]
        orientation = agent_state[2]
        status = agent_state[3]
        goal_x, goal_y = agent_state[4], agent_state[5]

        # Rotate the state based on goal position to ensure goal is bottom right
        if not is_right and not is_bottom:  # Goal is in top-left, rotate 180°
            # print("\nGoal is in top-left, rotate 180°")
            # Flip coordinates
            new_x = grid_size - x
            new_y = grid_size - y
            new_goal_x = grid_size - goal_x
            new_goal_y = grid_size - goal_y
            # Adjust orientation (rotate 180°)
            # print(f"{new_x=}, {new_y=}, {new_goal_x=}, {new_goal_y=}, {grid_size=}")
            new_orientation = (orientation + 2) % 4

        elif (
            not is_right and is_bottom
        ):  # Goal is in bottom-left, rotate 90° counter clockwise
            # print("\nGoal is in bottom-left, rotate 90° counter clockwise")
            # Swap and flip coordinates
            new_x = grid_size - y
            new_y = x
            new_goal_x = grid_size - goal_y
            new_goal_y = goal_x
            # Adjust orientation (rotate 90° clockwise)
            new_orientation = (orientation + 1) % 4

        elif (
            is_right and not is_bottom
        ):  # Goal is in top-right, rotate 90° clockwise
            # Swap and flip coordinates
            # print("\nGoal is in top-right, rotate 90° clockwise")
            new_x = y
            new_y = grid_size - x
            new_goal_x = goal_y
            new_goal_y = grid_size - goal_x
            # Adjust orientation (rotate 90° counter-clockwise)
            new_orientation = (orientation + 3) % 4
        else:
            # print("bottom-righ no change needed")
            new_x = x
            new_y = y
            new_goal_x = goal_x
            new_goal_y = goal_y
            new_orientation = orientation

        # Create the transformed agent state
        New_agent_state = np.concatenate(
            (
                [new_x, new_y, new_orientation, status, new_goal_x, new_goal_y],
                agent_state[6:12],
            ),
            axis=0,
        )
        # print(len(New_agent_state))
        # print("Transform state : ", New_agent_state)
        if len(agent_state) == 42:
            # print("long agent2 : ", len(agent_state[12:22]))
            agent_state_2 = rotate_agent_other_state(
                agent_state[12:22], is_right, is_bottom, grid_size
            )
            agent_state_3 = rotate_agent_other_state(
                agent_state[22:32], is_right, is_bottom, grid_size
            )
            agent_state_4 = rotate_agent_other_state(
                agent_state[32:], is_right, is_bottom, grid_size
            )
            complete_new_state_agent = np.concatenate(
                (New_agent_state, agent_state_2, agent_state_3, agent_state_4),
                axis=0,
            )
            # print("final length state agent: ", len(complete_new_state_agent))
        if len(agent_state) == 22:
            agent_state_2 = rotate_agent_other_state(
                agent_state[12:22], is_right, is_bottom, grid_size
            )
            complete_new_state_agent = np.concatenate(
                (New_agent_state, agent_state_2), axis=0
            )
        if len(agent_state) == 32:
            agent_state_2 = rotate_agent_other_state(
                agent_state[12:22], is_right, is_bottom, grid_size
            )
            agent_state_3 = rotate_agent_other_state(
                agent_state[32:], is_right, is_bottom, grid_size
            )
            complete_new_state_agent = np.concatenate(
                (New_agent_state, agent_state_2, agent_state_3), axis=0
            )
        complete_new_state[agent_idx] = complete_new_state_agent
    return complete_new_state


def extract_central_state(
        states: torch.Tensor, env, max_grid_size: int, device
    ) -> torch.Tensor:
    batch_size = 1
    central_state = torch.zeros(
        (batch_size, 1, max_grid_size, max_grid_size), device=device
    )
    dynamic_matrix = central_state_dynamic(
        states,
        env.grid_size,
        env.agent_positions,
        env.dynamic_obstacles,
        env.evacuated_agents,
        env.deactivated_agents,
        env.communication_range,
        env.goal_area,
        device,
    )
    # static_matrix = torch.ones(
    #     (grid_size, grid_size), device=device, dtype=torch.float32
    # ) 
    # Merging both dyanmic and static
    merged_matrix = torch.zeros_like(dynamic_matrix, device=device).float()
    # Create masks for different conditions
    both_positive = (dynamic_matrix > 0) & (static_matrix > 0)
    # save_tensor_as_markdown(both_positive, "both_positive.md")
    # print(both_positive)
    other_cases = ~both_positive  # Everything else
    
    # Apply the merging rules
    merged_matrix[both_positive] = (
        dynamic_matrix[both_positive] + static_matrix[both_positive]
    )
    merged_matrix[other_cases] = torch.min(
        dynamic_matrix[other_cases], static_matrix[other_cases]
    )
    
    # Assign to central state and return
    central_state[:, 0] = merged_matrix
    # save_tensor_as_markdown(merged_matrix, "merged.md")
    return central_state.flatten()
