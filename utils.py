import numpy as np

def riemann_covariance(features):
    cov = np.cov(features.T)  
    cov += np.eye(cov.shape[0]) * 1e-6

    eigvals, eigvecs = np.linalg.eigh(cov)  
    eigvals = np.clip(eigvals, 1e-12, None)


    log_cov = eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T
    return log_cov 

def euclidean_distance(cov, mean_cov_vec, cov_inv=None):
    vec = cov[np.triu_indices_from(cov)]
    diff = vec - mean_cov_vec
    
    if cov_inv is not None:
        return np.sqrt(diff.T @ cov_inv @ diff)
    else:
        return np.sqrt(diff.T @ diff)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def trial_to_cov_vec(trial: np.ndarray) -> np.ndarray:
    cov = riemann_covariance(trial)
    vec = cov[np.triu_indices_from(cov)]
    return vec / (np.linalg.norm(vec) + 1e-9)

