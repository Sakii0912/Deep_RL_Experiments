"""Empirical transition and reward models using non-parametric storage.

Instead of neural networks, these models store empirical estimates of:
- transition_model: maps (state, action) -> (next_state, done)
- reward_model: maps (state, action) -> reward

These are initialized from bootstrap data and updated as new trajectories arrive.
"""

import numpy as np
from collections import defaultdict
import torch


class EmpiricalTransitionModel:
    """Non-parametric transition model: Ψ(s_t, a_t) -> (s_{t+1}, d_t).
    
    Stores empirical estimates as (state, action) -> list of (next_state, done) pairs.
    When queried on unseen states, returns the average of stored transitions for that action.
    """
    
    def __init__(self, state_dim=3, action_dim=1, default_action_dim=4):
        """Initialize the empirical transition model.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (for interface compatibility)
            default_action_dim: Number of discrete actions (used if action_dim=1)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.default_action_dim = default_action_dim
        
        # Store empirical transitions: {(state_tuple, action): [(next_state, done), ...]}
        self.transitions = defaultdict(list)
        
        # Store running averages for fallback
        self.avg_next_state = {}
        self.avg_done = {}
        
    def update(self, states, actions, next_states, dones):
        """Add new transitions to the empirical model.
        
        Args:
            states: (N, state_dim) array
            actions: (N, action_dim) array
            next_states: (N, state_dim) array
            dones: (N,) array
        """
        for i in range(len(states)):
            state_tuple = tuple(states[i].astype(np.float32))
            action_val = int(actions[i, 0]) if actions.ndim > 1 else int(actions[i])
            key = (state_tuple, action_val)
            
            next_state = next_states[i].astype(np.float32)
            done = float(dones[i])
            
            self.transitions[key].append((next_state, done))
            self.avg_next_state[key] = np.mean([t[0] for t in self.transitions[key]], axis=0)
            self.avg_done[key] = np.mean([t[1] for t in self.transitions[key]])
    
    def forward(self, states, actions):
        """Query the transition model.
        
        Args:
            states: (B, state_dim) tensor
            actions: (B, action_dim) tensor
        
        Returns:
            (next_states, done_logits) where done_logits can be thresholded at 0.5
        """
        batch_size = states.shape[0]
        device = states.device
        
        next_states_list = []
        done_logits_list = []
        
        states_np = states.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy()
        
        for i in range(batch_size):
            state_tuple = tuple(states_np[i].astype(np.float32))
            action_val = int(actions_np[i, 0]) if actions_np.ndim > 1 else int(actions_np[i])
            key = (state_tuple, action_val)
            
            if key in self.avg_next_state:
                next_state = self.avg_next_state[key]
                done_prob = self.avg_done[key]
            else:
                # Fallback: if no data for this (state, action), return state unchanged
                next_state = states_np[i].astype(np.float32)
                done_prob = 0.0
            
            next_states_list.append(next_state)
            # Convert probability to logit (approximate)
            done_logit = np.log(done_prob + 1e-6) - np.log(1 - done_prob + 1e-6)
            done_logits_list.append(done_logit)
        
        next_states_tensor = torch.tensor(np.array(next_states_list), dtype=torch.float32, device=device)
        done_logits_tensor = torch.tensor(np.array(done_logits_list), dtype=torch.float32, device=device)
        
        return next_states_tensor, done_logits_tensor
    
    def parameters(self):
        """Return empty parameter list (for optimizer compatibility)."""
        return []
    
    def train(self):
        """No-op for compatibility with neural net interface."""
        pass
    
    def eval(self):
        """No-op for compatibility with neural net interface."""
        pass


class EmpiricalRewardModel:
    """Non-parametric reward model: Φ(s_t, a_t) -> r_t.
    
    Stores empirical estimates as (state, action) -> list of rewards.
    When queried on unseen states, returns the average stored reward for that action.
    """
    
    def __init__(self, state_dim=3, action_dim=1, default_action_dim=4):
        """Initialize the empirical reward model.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (for interface compatibility)
            default_action_dim: Number of discrete actions (used if action_dim=1)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.default_action_dim = default_action_dim
        
        # Store empirical rewards: {(state_tuple, action): [reward, ...]}
        self.rewards = defaultdict(list)
        self.avg_reward = {}
    
    def update(self, states, actions, reward_targets):
        """Add new reward observations to the empirical model.
        
        Args:
            states: (N, state_dim) array
            actions: (N, action_dim) array
            reward_targets: (N,) array
        """
        for i in range(len(states)):
            state_tuple = tuple(states[i].astype(np.float32))
            action_val = int(actions[i, 0]) if actions.ndim > 1 else int(actions[i])
            key = (state_tuple, action_val)
            
            reward = float(reward_targets[i])
            self.rewards[key].append(reward)
            self.avg_reward[key] = np.mean(self.rewards[key])
    
    def forward(self, states, actions):
        """Query the reward model.
        
        Args:
            states: (B, state_dim) tensor
            actions: (B, action_dim) tensor
        
        Returns:
            (rewards,) where rewards is (B,) shaped
        """
        batch_size = states.shape[0]
        device = states.device
        
        rewards_list = []
        
        states_np = states.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy()
        
        for i in range(batch_size):
            state_tuple = tuple(states_np[i].astype(np.float32))
            action_val = int(actions_np[i, 0]) if actions_np.ndim > 1 else int(actions_np[i])
            key = (state_tuple, action_val)
            
            if key in self.avg_reward:
                reward = self.avg_reward[key]
            else:
                # Fallback: return 1.0 (expected reward per step)
                reward = 1.0
            
            rewards_list.append(reward)
        
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        return rewards_tensor
    
    def parameters(self):
        """Return empty parameter list (for optimizer compatibility)."""
        return []
    
    def train(self):
        """No-op for compatibility with neural net interface."""
        pass
    
    def eval(self):
        """No-op for compatibility with neural net interface."""
        pass
