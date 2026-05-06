import numpy as np
from tqdm import tqdm

class EmpiricalAlgorithms:
    def __init__(self, env, alpha=0.9, n_samples=1000, q_samples=100, horizon=50):
        self.env = env
        self.alpha = alpha
        self.n = n_samples
        self.q = q_samples
        self.horizon = horizon

        self.player_range = range(1, 22)
        self.dealer_range = range(1, 11)
        self.actions = [0, 1]

    def _get_state_idx(self, p, d):
        return (p - 1, d - 1)

    def empirical_value_iteration(self, iterations=1000):
        V = np.zeros((21, 10))
        deltas = []

        for k in range(iterations):
            if hasattr(self.env, 'begin_next_buffer'):
                self.env.begin_next_buffer()

            V_new = np.zeros_like(V)
            for p in self.player_range:
                for d in self.dealer_range:
                    q_values = []
                    for a in self.actions:
                        total_return = 0
                        for _ in range(self.n):
                            next_p, next_d, r, term = self.env.step(p, d, a)
                            if not term:
                                total_return += r + self.alpha * V[self._get_state_idx(next_p, next_d)]
                            else:
                                total_return += r
                        q_values.append(total_return / self.n)

                    V_new[self._get_state_idx(p, d)] = np.max(q_values)

            if hasattr(self.env, 'commit_next_buffer'):
                self.env.commit_next_buffer()

            diff = np.max(np.abs(V_new - V))
            deltas.append(diff)
            V = V_new

        return V, deltas

    def empirical_policy_iteration(self, iterations=10):
        # Initialize with a greedy policy
        policy = np.zeros((21, 10), dtype=int)
        for p in self.player_range:
            if p < 17: policy[self._get_state_idx(p, 1)] = 0

        V = np.zeros((21, 10))
        deltas = []

        for k in range(iterations):
            if hasattr(self.env, 'begin_next_buffer'):
                self.env.begin_next_buffer()

            V_pi = np.zeros((21, 10))
            for p in self.player_range:
                for d in self.dealer_range:
                    total_path_return = 0
                    for _ in range(self.q):
                        curr_p, curr_d = p, d
                        path_reward = 0
                        for t in range(self.horizon):
                            action = policy[self._get_state_idx(curr_p, curr_d)]
                            curr_p, curr_d, r, term = self.env.step(curr_p, curr_d, action)
                            path_reward += (self.alpha ** t) * r
                            if term: break
                        total_path_return += path_reward
                    V_pi[self._get_state_idx(p, d)] = total_path_return / self.q

            new_policy = np.zeros_like(policy)
            for p in self.player_range:
                for d in self.dealer_range:
                    q_values = []
                    for a in self.actions:
                        total_val = 0
                        for _ in range(self.n):
                            next_p, next_d, r, term = self.env.step(p, d, a)
                            if not term:
                                total_val += r + self.alpha * V_pi[self._get_state_idx(next_p, next_d)]
                            else:
                                total_val += r
                        q_values.append(total_val / self.n)
                    new_policy[self._get_state_idx(p, d)] = np.argmax(q_values)

            if hasattr(self.env, 'commit_next_buffer'):
                self.env.commit_next_buffer()

            diff = np.max(np.abs(V_pi - V))
            deltas.append(diff)
            V = V_pi
            policy = new_policy

        return V, policy, deltas
