import numpy as np
from utils import cosine_sim

class RiemannAuthenticator:
    """
    Nearest-centroid classifier in Riemannian covariance space.

    Threshold tau is estimated as the lower-tail percentile of genuine
    cosine similarity scores computed on the enrolment set itself.
    """

    def __init__(self, threshold_percentile: float = 10.0):
        self.threshold_percentile = threshold_percentile
        self.templates: dict[int, np.ndarray] = {}
        self.tau: float = 0.0

    def fit(self, X_cov: np.ndarray, y: np.ndarray) -> None:
        """
        X_cov : (N, D) normalised covariance vectors
        y     : (N,)  integer class labels
        """
        for cls in np.unique(y):
            tmpl = X_cov[y == cls].mean(axis=0)
            self.templates[int(cls)] = tmpl / (np.linalg.norm(tmpl) + 1e-9)

        genuine_scores = []
        for cls in np.unique(y):
            tmpl = self.templates[int(cls)]
            for v in X_cov[y == cls]:
                genuine_scores.append(cosine_sim(v, tmpl))

        self.tau = float(np.percentile(genuine_scores, self.threshold_percentile))

    def verify(self, cov_vec: np.ndarray,
               claimed_id: int) -> tuple[bool, float]:
        score = cosine_sim(cov_vec, self.templates[claimed_id])
        return score >= self.tau, score