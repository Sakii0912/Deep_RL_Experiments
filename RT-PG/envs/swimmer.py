"""
Swimmer environment wrapper for uniform RL experiments.
"""
import gymnasium as gym
import numpy as np


class Swimmer:
    """
    Swimmer-v4 environment wrapper.
    
    State: [position, velocity] for 3 body parts (8D state)
    Action space: continuous 2D action for joint torques
    """
    
    def __init__(self):
        self.env = gym.make("Swimmer-v4")
        self.observation = None
        
    def reset(self):
        """Reset environment and return initial observation."""
        self.observation, info = self.env.reset()
        return self.observation.copy(), info
    
    def step(self, action):
        """
        Take one step in the environment.
        
        Args:
            action: np.ndarray of shape (2,) with continuous values in [-1, 1]
            
        Returns:
            observation: np.ndarray of shape (8,)
            reward: float
            terminated: bool
            truncated: bool
            info: dict
        """
        self.observation, reward, terminated, truncated, info = self.env.step(action)
        return self.observation.copy(), reward, terminated, truncated, info
    
    @property
    def action_space(self):
        """Get action space."""
        return self.env.action_space
    
    @property
    def observation_space(self):
        """Get observation space."""
        return self.env.observation_space
    
    def close(self):
        """Close the environment."""
        self.env.close()
