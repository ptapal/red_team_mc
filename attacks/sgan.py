import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import trial_to_cov_vec, cosine_sim

# hyperparameters

LATENT_DIM   = 32
HIDDEN_DIM   = 64
S_HIDDEN_DIM = 128
D_HIDDEN_DIM = 64
LR_G         = 3e-4
LR_D         = 3e-4
LR_S         = 5e-4
BETA_KL      = 1e-3
LAMBDA_D     = 0.3    # weight of D's loss in G's objective
S_EPOCHS     = 40
MAX_S_PAIRS  = 3000   # cap on training pairs; prevents 5+ GB tensors for MI


class Surrogate(nn.Module):
    def __init__(self, input_dim: int, template_dim: int):
        super().__init__()
        in_dim = input_dim + template_dim
        hd = S_HIDDEN_DIM
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hd),   nn.ELU(),
            nn.Linear(hd, hd),       nn.ELU(),
            nn.Linear(hd, hd // 2), nn.ELU(),
            nn.Linear(hd // 2, 1),
        )

    def forward(self, x_flat: torch.Tensor,
                mu_t: torch.Tensor) -> torch.Tensor:
        if mu_t.dim() == 1:
            mu_t = mu_t.unsqueeze(0).expand(x_flat.shape[0], -1)
        return self.net(torch.cat([x_flat, mu_t], dim=-1)).squeeze(-1)


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        hd = HIDDEN_DIM
        self.norm = nn.LayerNorm(input_dim)
        self.net  = nn.Sequential(
            nn.Linear(input_dim, hd), nn.ELU(),
            nn.Linear(hd, hd),        nn.ELU(),
        )
        self.mu_head = nn.Linear(hd, latent_dim)
        self.lv_head = nn.Linear(hd, latent_dim)

    def forward(self, x):
        h = self.net(self.norm(x))
        return self.mu_head(h), self.lv_head(h)


class _Decoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        hd = HIDDEN_DIM
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hd), nn.ELU(),
            nn.Linear(hd, hd),          nn.ELU(),
            nn.Linear(hd, input_dim),
        )

    def forward(self, z):
        return self.net(z)


class VAEGenerator(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = _Encoder(input_dim, latent_dim)
        self.decoder = _Decoder(input_dim, latent_dim)

    def forward(self, x_flat: torch.Tensor):
        mu, lv = self.encoder(x_flat)
        lv      = lv.clamp(-6, 2)
        std     = (0.5 * lv).exp()
        z       = mu + std * torch.randn_like(std)   # reparameterisation -- differentiable
        rms     = float(x_flat.norm().item() / math.sqrt(x_flat.numel()) + 1e-9)
        delta   = self.decoder(z) * rms
        return delta, mu, lv


class Discriminator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        hd = D_HIDDEN_DIM
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hd), nn.ELU(),
            nn.Linear(hd, hd // 2),   nn.ELU(),
            nn.Linear(hd // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _kl(mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())


def train_surrogate(
    templates: dict,
    X_enroll: list,
    y_enroll: np.ndarray,
    device: str = "cpu",
    n_epochs: int = S_EPOCHS,
    seed: int = 42,
    score_fn=None,
) -> "Surrogate":
    torch.manual_seed(seed)
    classes      = list(templates.keys())
    input_dim    = int(np.prod(X_enroll[0].shape))
    template_dim = len(next(iter(templates.values())))

    cov_vecs, valid_X = [], []
    for x in X_enroll:
        try:
            cov_vecs.append(trial_to_cov_vec(x))
            valid_X.append(x)
        except Exception:
            pass

    xs_list, mus_list, scores_list = [], [], []
    for i, (q, x) in enumerate(zip(cov_vecs, valid_X)):
        x_flat = x.flatten().astype(np.float32)
        for k in classes:
            s = (cosine_sim(q, templates[k]) if score_fn is None
                 else float(score_fn(q, k)))
            xs_list.append(x_flat)
            mus_list.append(templates[k].astype(np.float32))
            scores_list.append(float(s))

    if not xs_list:
        return Surrogate(input_dim, template_dim).to(device)

    if len(xs_list) > MAX_S_PAIRS:
        rng_sub = np.random.default_rng(seed)
        idx_sub = rng_sub.choice(len(xs_list), MAX_S_PAIRS, replace=False)
        xs_list    = [xs_list[i]    for i in idx_sub]
        mus_list   = [mus_list[i]   for i in idx_sub]
        scores_list= [scores_list[i] for i in idx_sub]

    xs_t   = torch.tensor(np.stack(xs_list),   dtype=torch.float32, device=device)
    mus_t  = torch.tensor(np.stack(mus_list),  dtype=torch.float32, device=device)
    tgt_t  = torch.tensor(scores_list,          dtype=torch.float32, device=device)

    S     = Surrogate(input_dim, template_dim).to(device)
    opt_S = torch.optim.Adam(S.parameters(), lr=LR_S)

    n, bs = len(xs_t), min(256, len(xs_t))
    for _ in range(n_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, bs):
            idx_b = perm[start:start + bs]
            loss  = F.mse_loss(S(xs_t[idx_b], mus_t[idx_b]), tgt_t[idx_b])
            opt_S.zero_grad()
            loss.backward()
            opt_S.step()

    with torch.no_grad():
        pred = S(xs_t, mus_t).cpu().numpy()
    true = np.asarray(scores_list, dtype=float)
    denom = float(pred.std() * true.std())
    S.fidelity = dict(
        r   = float(np.corrcoef(pred, true)[0, 1]) if denom > 1e-12 else float("nan"),
        mse = float(np.mean((pred - true) ** 2)),
        n   = int(len(true)),
    )

    S.eval()
    return S


def run_surrogate_gan_attack(
    templates: dict,
    tau: float,
    X_test: list,
    y_test: np.ndarray,
    X_enroll: list,
    y_enroll: np.ndarray,
    n_iter: int,
    rng: np.random.Generator,
    lam: float = 0.3,
    device: str = "cpu",
    seed: int = 42,
    score_fn=None,
    surrogate_score_fn=None,
) -> tuple[list, dict, "Discriminator", "Surrogate", "VAEGenerator"]:
    torch.manual_seed(seed)

    X_arr   = np.array(X_test)
    classes = np.array(list(templates.keys()))
    logs    = []
    per_tgt = {int(c): [] for c in classes}

    input_dim    = int(np.prod(X_arr.shape[1:]))
    S = train_surrogate(templates, X_enroll, y_enroll, device=device, seed=seed,
                        score_fn=(surrogate_score_fn if surrogate_score_fn
                                  is not None else score_fn))
    S.eval()
    for param in S.parameters():
        param.requires_grad_(False)

    G     = VAEGenerator(input_dim, LATENT_DIM).to(device)
    D     = Discriminator(input_dim).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)

    genuine_flat = [
        torch.tensor(x.flatten(), dtype=torch.float32, device=device)
        for x in X_arr
    ]
    ones  = torch.ones(1,  device=device)
    zeros = torch.zeros(1, device=device)

    for _ in range(n_iter):
        vi     = int(rng.integers(len(X_arr)))
        src    = int(y_test[vi])
        others = [c for c in classes if c != src]
        if not others:
            continue
        tgt = int(rng.choice(others))
        if tgt not in templates:
            continue

        x_flat = genuine_flat[vi].unsqueeze(0)
        mu_tgt = torch.tensor(templates[tgt], dtype=torch.float32, device=device)

        G.train()
        delta, mu_z, lv_z = G(x_flat)
        x_prime = x_flat + delta            

        s_score = S(x_prime, mu_tgt)        

        D.train()
        d_logit_fake = D(x_prime)

        budget_t = delta.norm() / (x_flat.norm() + 1e-9)

        loss_G = (
            -(1 - lam) * s_score
            +  lam     * budget_t
            + LAMBDA_D * F.binary_cross_entropy_with_logits(d_logit_fake, ones)
            + BETA_KL  * _kl(mu_z, lv_z)
        )
        opt_G.zero_grad()
        loss_G.backward()
        nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt_G.step()

        gi         = int(rng.integers(len(genuine_flat)))
        x_gen_t    = genuine_flat[gi].unsqueeze(0)
        x_fake_det = x_prime.detach()

        d_loss = (
            F.binary_cross_entropy_with_logits(D(x_gen_t), ones)
          + F.binary_cross_entropy_with_logits(D(x_fake_det), zeros)
        )
        opt_D.zero_grad()
        d_loss.backward()
        nn.utils.clip_grad_norm_(D.parameters(), 1.0)
        opt_D.step()

        X0       = X_arr[vi]
        xp_np    = x_prime.detach().cpu().numpy().reshape(X0.shape)
        try:
            q_adv = trial_to_cov_vec(xp_np)
        except Exception:
            continue

        real_score = (cosine_sim(q_adv, templates[tgt]) if score_fn is None
                      else float(score_fn(q_adv, tgt)))
        accepted   = bool(real_score >= tau)
        budget_np  = float(np.linalg.norm(xp_np - X0) / (np.linalg.norm(X0) + 1e-9))

        try:
            q0 = trial_to_cov_vec(X0)
            score0 = (cosine_sim(q0, templates[tgt]) if score_fn is None
                      else float(score_fn(q0, tgt)))
            accepted0 = bool(score0 >= tau)
        except Exception:
            accepted0 = None

        logs.append(dict(
            accepted = accepted,
            s_score  = float(s_score.detach()),
            d_score  = float(torch.sigmoid(d_logit_fake).detach()),
            budget   = budget_np,
            accepted_zero_effort = accepted0,
        ))
        per_tgt[tgt].append(accepted)

    return logs, per_tgt, D, S, G


def generate_perturbed(
    G: "VAEGenerator",
    X_trials: list,
    device: str = "cpu",
    seed: int = 0,
) -> list:
    torch.manual_seed(seed)
    G.eval()
    perturbed = []
    with torch.no_grad():
        for x in X_trials:
            x_flat = torch.tensor(
                x.flatten(), dtype=torch.float32, device=device
            ).unsqueeze(0)
            delta, _, _ = G(x_flat)
            xp = (x_flat + delta).squeeze(0).cpu().numpy().reshape(x.shape)
            perturbed.append(xp)
    return perturbed


def eval_discriminator(
    D: "Discriminator",
    X_genuine: list,
    X_perturbed: list,
    device: str = "cpu",
    threshold: float = 0.5,
) -> dict:
    """Returns dict: tpr, tnr, auroc, mean_d_genuine, mean_d_perturbed."""
    D.eval()

    def _score(arrays):
        out = []
        with torch.no_grad():
            for x in arrays:
                xt = torch.tensor(
                    x.flatten(), dtype=torch.float32, device=device
                ).unsqueeze(0)
                out.append(float(torch.sigmoid(D(xt)).item()))
        return np.array(out)

    s_real = _score(X_genuine)
    s_fake = _score(X_perturbed)

    tpr = float((s_real >= threshold).mean())
    tnr = float((s_fake <  threshold).mean())

    labels = np.concatenate([np.ones(len(s_real)), np.zeros(len(s_fake))])
    scores = np.concatenate([s_real, s_fake])
    order  = np.argsort(-scores)
    tpr_c  = np.cumsum(labels[order])      / max(labels.sum(), 1)
    fpr_c  = np.cumsum(1 - labels[order])  / max((1 - labels).sum(), 1)
    auroc  = float(abs(np.trapz(tpr_c, fpr_c)))

    return dict(
        tpr              = tpr,
        tnr              = tnr,
        auroc            = auroc,
        mean_d_genuine   = float(s_real.mean()),
        mean_d_perturbed = float(s_fake.mean()),
    )
