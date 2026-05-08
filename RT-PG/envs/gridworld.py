"""Simple discrete gridworld environment for RL experiments.

The gridworld is an NxN grid where the agent:
- Starts at position (0, 0)
- Can move up, down, left, right (4 discrete actions)
- Goal is to reach position (goal_x, goal_y)
- Receives reward of +1 per step until goal, then episode terminates
- Episodes terminate when goal is reached or max_steps is exceeded
"""

import numpy as np


class DiscreteGridWorldEnv:
    """Simple NxN discrete gridworld environment."""
    
    def __init__(self, grid_size=5, max_steps=50, goal_x=None, goal_y=None):
        """Initialize gridworld.
        
        Args:
            grid_size: Size of the grid (NxN)
            max_steps: Maximum steps per episode
            goal_x: X coordinate of goal (default: top-right)
            goal_y: Y coordinate of goal (default: top-right)
        """
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.goal_x = goal_x if goal_x is not None else grid_size - 1
        self.goal_y = goal_y if goal_y is not None else grid_size - 1
        
        # State: (x, y, steps_taken)
        self.x = 0
        self.y = 0
        self.steps = 0
        self.done = False
        
        # Action mapping: 0=up, 1=down, 2=left, 3=right
        self.action_meanings = ['up', 'down', 'left', 'right']
        self.num_actions = 4
        
    def reset(self, seed=None):
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        self.x = 0
        self.y = 0
        self.steps = 0
        self.done = False
        return self._get_state()
    
    def _get_state(self):
        """Return current state as [x, y, steps]."""
        return np.array([self.x, self.y, self.steps], dtype=np.float32)
    
    def step(self, action):
        """Take one step in the environment.
        
        Args:
            action: Integer in [0, 1, 2, 3] representing direction
        
        Returns:
            (state, reward, done, truncated, info)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")
        
        # Convert action to movement
        dx, dy = 0, 0
        if action == 0:  # up
            dy = -1
        elif action == 1:  # down
            dy = 1
        elif action == 2:  # left
            dx = -1
        elif action == 3:  # right
            dx = 1
        else:
            raise ValueError(f"Invalid action: {action}")
        
        # Apply movement with boundary clamping
        self.x = np.clip(self.x + dx, 0, self.grid_size - 1)
        self.y = np.clip(self.y + dy, 0, self.grid_size - 1)
        self.steps += 1
        
        # Check if goal reached
        reached_goal = (self.x == self.goal_x and self.y == self.goal_y)
        truncated = (self.steps >= self.max_steps)
        self.done = reached_goal or truncated
        
        # Reward: +1 for each step toward goal or at goal
        reward = 1.0
        
        next_state = self._get_state()
        
        return next_state, reward, reached_goal, truncated, {
            "reached_goal": reached_goal,
            "position": (self.x, self.y),
            "steps": self.steps
        }
    
    def seed(self, seed=None):
        """Seed the random number generator."""
        np.random.seed(seed)
        return [seed]
    
    @property
    def observation_space(self):
        """Return observation space shape."""
        class Shape:
            def __init__(self, shape_tuple):
                self.shape = shape_tuple
        return Shape((3,))  # [x, y, steps]
    
    @property
    def action_space(self):
        """Return action space shape."""
        class Shape:
            def __init__(self, shape_tuple):
                self.shape = shape_tuple
        return Shape((1,))  # discrete action as 1D array
    
    def close(self):
        """Clean up environment resources."""
        pass
