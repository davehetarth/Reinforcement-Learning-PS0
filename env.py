import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ApartmentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, T: int, K: int, noise_std: float = 0.0, seed: int = None):
        super().__init__()
        self.T = T
        self.K = K
        self.noise_std = noise_std
        
        # Action space: 0 = reject, 1 = accept
        self.action_space = spaces.Discrete(2)
        
        # Observation space: Box(low=[1.0, -inf], high=[T, inf], dtype=float32)
        # Handles continuous noisy observations comfortably
        low = np.array([1.0, -np.inf], dtype=np.float32)
        high = np.array([float(T), np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self._np_random = None
        if seed is not None:
            self.reset(seed=seed)

    def _get_obs(self):
        if self.noise_std > 0.0:
            epsilon = self.np_random.normal(0, self.noise_std)
            u_hat = self.current_U + epsilon
        else:
            u_hat = float(self.current_U)
        return np.array([float(self.current_t), u_hat], dtype=np.float32)

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        
        self.current_t = 1
        self.current_U = self.np_random.integers(1, self.K + 1)
        
        return self._get_obs(), {}

    def step(self, action: int):
        # Terminated: Accepted an apartment or rejected at the final step T
        # Truncated: Not explicitly used here but required by Gymnasium API
        terminated = False
        truncated = False
        reward = 0.0
        
        if action == 1:  # ACCEPT
            reward = float(self.current_U)
            terminated = True
        else:  # REJECT
            if self.current_t >= self.T:
                reward = 0.0  # Fallback to subletting
                terminated = True
            else:
                self.current_t += 1
                self.current_U = self.np_random.integers(1, self.K + 1)
                
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}