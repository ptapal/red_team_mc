import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import math
import numpy as np
import torch
import torch.nn as nn

from utils import trial_to_cov_vec, cosine_sim

# hyperparameters
LATENT_DIM     = 32
HIDDEN_DIM     = 256       # used when input_dim <= 10_000
HIDDEN_DIM_BIG = 128       # used for MI trials (input_dim > 10_000)
LR             = 3e-4
BETA_KL        = 1e-3      # KL weight (light regularisation)
BASELINE_ALPHA = 0.05      # EWA update coefficient for running baseline


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.net  = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.lv_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(self.norm(x))
        return self.mu_head(h), self.lv_head(h)


class _Decoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAEGenerator(nn.Module):

    def __init__(self, input_dim: int,
                 latent_dim: int = LATENT_DIM):
        super().__init__()
        hd = HIDDEN_DIM_BIG if input_dim > 10_000 else HIDDEN_DIM
        self.encoder = _Encoder(input_dim, latent_dim, hd)
        self.decoder = _Decoder(input_dim, latent_dim, hd)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor, rms: torch.Tensor):
        mu, lv = self.encoder(x)
        lv     = lv.clamp(-6, 2)                     
        std    = (0.5 * lv).exp()
        z      = mu + std * torch.randn_like(mu)
        delta  = self.decoder(z) * rms              
        return delta, mu, lv, z


def _log_q(z: torch.Tensor,
           mu: torch.Tensor,
           lv: torch.Tensor) -> torch.Tensor:
    var = lv.exp() + 1e-8
    return -0.5 * torch.sum((z - mu).pow(2) / var + lv + math.log(2 * math.pi))


def _kl(mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())


def run_vae_attack(templates: dict,
                   tau: float,
                   X_test: list,
                   y_test: np.ndarray,
                   n_iter: int,
                   rng: np.random.Generator,
                   lam: float = 0.3,
                   device: str = "cpu",
                   seed: int = 42,
                   score_fn=None) -> tuple[list, dict]:
    torch.manual_seed(seed)

    X_arr   = np.array(X_test)
    classes = np.array(list(templates.keys()))
    logs    = []
    per_tgt = {int(c): [] for c in classes}

    input_dim = int(np.prod(X_arr.shape[1:]))
    model     = VAEGenerator(input_dim, latent_dim=LATENT_DIM).to(device)
    opt       = torch.optim.Adam(model.parameters(), lr=LR)

    baseline = 0.0

    for _ in range(n_iter):
        vi   = int(rng.integers(len(X_arr)))
        X0   = X_arr[vi]
        src  = int(y_test[vi])
        others = [c for c in classes if c != src]
        if not others:
            continue
        tgt = int(rng.choice(others))
        if tgt not in templates:
            continue

        x_flat = torch.tensor(X0.flatten(), dtype=torch.float32,
                               device=device).unsqueeze(0)
        rms    = torch.tensor(
            float(np.linalg.norm(X0) / math.sqrt(X0.size) + 1e-9),
            dtype=torch.float32, device=device,
        )

        model.train()
        delta, mu, lv, z = model(x_flat, rms)
        delta_np = delta.squeeze(0).detach().cpu().numpy().reshape(X0.shape)
        Xp       = X0 + delta_np

        try:
            q = trial_to_cov_vec(Xp)
        except Exception:
            opt.zero_grad()
            continue

        score    = (cosine_sim(q, templates[tgt]) if score_fn is None
                    else float(score_fn(q, tgt)))
        accepted = bool(score >= tau)
        budget   = float(np.linalg.norm(Xp - X0) / (np.linalg.norm(X0) + 1e-9))

        try:
            q0 = trial_to_cov_vec(X0)
            score0 = (cosine_sim(q0, templates[tgt]) if score_fn is None
                      else float(score_fn(q0, tgt)))
            accepted0 = bool(score0 >= tau)
        except Exception:
            accepted0 = None
        reward   = (1 - lam) * float(accepted) + lam * max(0.0, 1.0 - budget)

        advantage = reward - baseline
        baseline += BASELINE_ALPHA * (reward - baseline)

        adv_t = torch.tensor(advantage, dtype=torch.float32, device=device)
        loss  = -adv_t * _log_q(z, mu, lv) + BETA_KL * _kl(mu, lv)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        logs.append(dict(accepted=accepted, reward=reward, budget=budget,
                         accepted_zero_effort=accepted0))
        per_tgt[tgt].append(accepted)

    return logs, per_tgt
