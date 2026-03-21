import numpy as np
import math

class UCBSelector:
    """UCB1 bandit. Each arm is pulled once before UCB scores are used."""

    def __init__(self, n_arms: int, c: float = math.sqrt(2)):
        self.n_arms  = n_arms
        self.c       = c
        self.counts  = np.zeros(n_arms, dtype=int)
        self.rewards = np.zeros(n_arms, dtype=float)

    @property
    def means(self) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return np.where(self.counts > 0,
                            self.rewards / self.counts, 0.0)

    def select(self) -> int:
        t = int(self.counts.sum())
        if t < self.n_arms:
            return t
        ucb = self.means + self.c * np.sqrt(np.log(t) / (self.counts + 1e-9))
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm]  += 1
        self.rewards[arm] += reward

    def ucb_scores(self) -> np.ndarray:
        t = int(self.counts.sum())
        return self.means + self.c * np.sqrt(np.log(max(t, 1)) / (self.counts + 1e-9))