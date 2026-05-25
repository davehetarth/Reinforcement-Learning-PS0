import numpy as np

class RandomPolicy:
    def __init__(self, T: int):
        self.T = T

    def act(self, obs: np.ndarray) -> int:
        t = int(obs[0])
        # Accept with probability 1/T each week
        prob_accept = 1.0 / self.T
        return int(np.random.rand() < prob_accept)


class ThresholdPolicy:
    def __init__(self, u_min: float):
        self.u_min = u_min

    def act(self, obs: np.ndarray) -> int:
        u_observed = obs[1]
        return 1 if u_observed >= self.u_min else 0


class OptimalPolicy:
    def __init__(self):
        # Hardcoded threshold lookup matching Problem 1(c) values of W_{t+1}
        # Keys are week 't'. Value is accepted if u >= threshold
        self.thresholds = {
            1: 3.25,
            2: 3.00,
            3: 2.50,
            4: 0.00
        }

    def act(self, obs: np.ndarray) -> int:
        t = int(obs[0])
        u_observed = obs[1]
        threshold = self.thresholds.get(t, 0.0)
        return 1 if u_observed >= threshold else 0