import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MinimalPINN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self.act = nn.Tanh()
        self.ln = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        out = self.act(self.fc1(x))
        out = self.ln(out)
        resid = out
        out = self.act(self.fc2(out)) + resid
        out = self.drop(out)
        out = self.fc3(out)
        return out

def preprocess_input(x):
    if isinstance(x, np.ndarray):
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        elif x.ndim == 2:
            pass
        elif x.ndim == 1:
            x = x.reshape(1, -1)
        x = (x - np.mean(x, axis=1, keepdims=True)) / (np.std(x, axis=1, keepdims=True) + 1e-8)
        return torch.tensor(x, dtype=torch.float32)
    elif torch.is_tensor(x):
        if x.dim() == 3:
            x = x.view(x.shape[0], -1)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        return (x - mean) / std
    else:
        raise ValueError("Unsupported input type")


def build_realistic_laplacian(ch_names, k=4, sigma_scale=1.0, use_mne=True):
    n = len(ch_names)
    pos = {}
    if use_mne:
        try:
            import mne
            mont = mne.channels.make_standard_montage('standard_1020')
            ch_pos = mont.get_positions()['ch_pos']
            for ch in ch_names:
                if ch in ch_pos:
                    pos[ch] = np.array(ch_pos[ch], dtype=float)
            if len(pos) < max(8, int(0.7 * n)):
                pos = {}
        except Exception:
            pos = {}

    if not pos:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        for i, ch in enumerate(ch_names):
            pos[ch] = np.array([np.cos(angles[i]), np.sin(angles[i]), 0.0], dtype=float)

    P = np.vstack([pos[ch] for ch in ch_names]) 
    dists = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    mean_d = np.mean(dists[np.triu_indices(n, k=1)]) if n > 1 else 1.0
    sigma = (mean_d * sigma_scale) if mean_d > 0 else 1.0

    W = np.exp(- (dists ** 2) / (2 * sigma**2))
    np.fill_diagonal(W, 0.0)

    if k is not None and k < n:
        W_sparse = np.zeros_like(W)
        for i in range(n):
            idxs = np.argsort(dists[i])
            neighbors = idxs[1:k+1]
            W_sparse[i, neighbors] = W[i, neighbors]
        W = np.maximum(W_sparse, W_sparse.T)

    deg = np.sum(W, axis=1)
    with np.errstate(divide='ignore'):
        d_inv_sqrt = 1.0 / np.sqrt(deg)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_sym = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
    return torch.tensor(L_sym, dtype=torch.float32)  

def compute_spatial_laplacian(u, L):
    b, c, t = u.shape
    ut = u.permute(0, 2, 1).contiguous().view(-1, c)  
    lap = torch.matmul(ut, L.T)                      
    lap = lap.view(b, t, c).permute(0, 2, 1)         
    return lap


def train_pinn_pde(model,
                   x_train, y_train,
                   ch_names,
                   epochs=200,
                   lr=1e-3,
                   laplace_lambda=0.1,
                   phys_lambda=0.5,
                   k=4,
                   sigma_scale=1.0,
                   D_init=0.5,
                   learn_D=False,
                   dt=1.0,
                   device=None,
                   print_every=25):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if isinstance(x_train, np.ndarray):
        x_t = preprocess_input(x_train).to(device)
    else:
        x_t = preprocess_input(x_train).to(device)
    if isinstance(y_train, np.ndarray):
        y_t = preprocess_input(y_train).to(device)
    else:
        y_t = preprocess_input(y_train).to(device)

    feat = x_t.shape[1]
    n_channels = len(ch_names)
    if feat % n_channels != 0:
        raise ValueError("feature dimension not divisible by n_channels")
    n_time = feat // n_channels

    L = build_realistic_laplacian(ch_names, k=k, sigma_scale=sigma_scale).to(device)

    if learn_D:
        D_param = nn.Parameter(torch.tensor(np.log(np.exp(D_init) - 1.0), dtype=torch.float32).to(device))
        opt_params = list(model.parameters()) + [D_param]
    else:
        D_fixed = float(D_init)
        D_param = None
        opt_params = model.parameters()

    optimizer = optim.Adam(opt_params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        y_pred = model(x_t)  
        b = y_pred.shape[0]
        u = y_pred.view(b, n_channels, n_time) 
        u_true = y_t.view(b, n_channels, n_time)

        mse_loss = nn.MSELoss()(y_pred, y_t)

        lap = compute_spatial_laplacian(u, L) 
        lap_loss = torch.mean(lap ** 2)

        if learn_D:
            D = torch.nn.functional.softplus(D_param)
        else:
            D = torch.tensor(D_fixed, dtype=torch.float32, device=device)

        if n_time < 2:
            phys_loss = torch.tensor(0.0, device=device)
        else:
            du_dt = (u[:, :, 1:] - u[:, :, :-1]) / float(dt)      
            lap_t = lap[:, :, :-1]                             
            resid = du_dt - D * lap_t
            phys_loss = torch.mean(resid ** 2)

        total_loss = mse_loss + laplace_lambda * lap_loss + phys_lambda * phys_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            D_val = D.item() if not learn_D else torch.nn.functional.softplus(D_param).item()
            print(f"epoch {epoch}/{epochs}  mse={mse_loss.item():.6e}  lap={lap_loss.item():.6e}  phys={phys_loss.item():.6e}  D={D_val:.4f}  total={total_loss.item():.6e}")

    learned_D = (torch.nn.functional.softplus(D_param).item() if learn_D else D_fixed)
    return model, learned_D