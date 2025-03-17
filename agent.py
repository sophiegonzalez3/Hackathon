import numpy as np

class MyAgents():
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