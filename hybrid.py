import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pinns import compute_spatial_laplacian

def hybrid_preprocess_input(x):
    if isinstance(x, np.ndarray):
        x = x.reshape(1, -1) if x.ndim == 1 else x
        x = (x - np.mean(x, axis=1, keepdims=True)) / (np.std(x, axis=1, keepdims=True) + 1e-8)
        return torch.tensor(x, dtype=torch.float32)
    elif torch.is_tensor(x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        return (x - mean) / std
    else:
        raise ValueError("Unsupported input type")

# no mne montage in use, unlike the ordinary pinns one
def hybrid_build_realistic_laplacian(features_np, k=4):
    # drop constant features before corrcoef
    std = features_np.std(axis=0)
    valid = std > 1e-6
    features_filtered = features_np[:, valid]
    
    if features_filtered.shape[1] < 2:
        # fallback: identity Laplacian (no regularisation)
        n = features_np.shape[1]
        return torch.eye(n, dtype=torch.float32)
    
    corr = np.corrcoef(features_filtered.T)
    corr = np.nan_to_num(corr, nan=0.0)
    corr = np.clip(corr, 0, None)
    np.fill_diagonal(corr, 0)
    W = np.zeros_like(corr)
    for i in range(corr.shape[0]):
        idxs = np.argsort(corr[i])[-k:]
        W[i, idxs] = corr[i, idxs]
    W = np.maximum(W, W.T)
    deg = W.sum(axis=1)
    d_inv_sqrt = np.diag(1 / np.sqrt(deg + 1e-8))
    L_small = np.eye(len(W)) - d_inv_sqrt @ W @ d_inv_sqrt
    
    # put back into full latent_dim x latent_dim matrix
    n = features_np.shape[1]
    L_full = np.eye(n)
    idx = np.where(valid)[0]
    for i, gi in enumerate(idx):
        for j, gj in enumerate(idx):
            L_full[gi, gj] = L_small[i, j]
    
    return torch.tensor(L_full, dtype=torch.float32)