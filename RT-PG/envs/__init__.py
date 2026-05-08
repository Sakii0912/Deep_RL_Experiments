"""
Uniform RL environment wrappers for RT-PG experiments.

All environments follow a consistent interface:
- reset() -> observation
- step(action) -> (observation, reward, terminated, truncated, info)
- action_space property
- observation_space property
"""

from .cartpole import ContinuousCartPoleEnv
from .halfcheetah import HalfCheetah
from .swimmer import Swimmer

__all__ = ["ContinuousCartPoleEnv", "HalfCheetah", "Swimmer"]
