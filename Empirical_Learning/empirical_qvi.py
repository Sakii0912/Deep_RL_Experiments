import numpy as np
from tqdm import tqdm

class EmpiricalQVI:
    def __init__(self, env, gamma=0.9, n_samples=100):
        self.env = env
        self.gamma = gamma
        self.n = n_samples

        self.Q = np.zeros((21, 10, 2))

    def _get_idx(self, p, d):
        return p - 1, d - 1

    def run(self, iterations=50):
        deltas = []

        for k in tqdm(range(iterations), desc="EQVI Iterations"):
            if hasattr(self.env, 'begin_next_buffer'):
                self.env.begin_next_buffer()

            Q_prev = self.Q.copy()
            Q_new = np.zeros_like(self.Q)

            for p in range(1, 22):
                for d in range(1, 11):
                    p_idx, d_idx = self._get_idx(p, d)

                    for a in [0, 1]:
                        sample_returns = []
                        for _ in range(self.n):
                            next_p, next_d, r, term = self.env.step(p, d, a)

                            if term:
                                sample_returns.append(r)
                            else:
                                next_p_idx, next_d_idx = self._get_idx(next_p, next_d)
                                next_val = np.max(Q_prev[next_p_idx, next_d_idx])
                                sample_returns.append(r + self.gamma * next_val)

                        Q_new[p_idx, d_idx, a] = np.mean(sample_returns)

            if hasattr(self.env, 'commit_next_buffer'):
                self.env.commit_next_buffer()

            diff = np.max(np.abs(Q_new - Q_prev))
            deltas.append(diff)
            self.Q = Q_new

        return self.Q, deltas
