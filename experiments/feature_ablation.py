"""
Feature space comparison: Riemannian log-cov vs. Euclidean cov vs. PSD.

Runs the UCB bandit attack at p=10 across all 8 datasets for each feature
type. Reports baseline FAR, EER, and adversarial FAR.

Output:
  results_feature_comparison/feature_comparison.csv
  results_feature_comparison/feature_comparison.png
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

warnings.filterwarnings("ignore")

from helpers import load_sessions
from utils   import cosine_sim, riemann_covariance
from arms    import AttackArms
from ucb     import UCBSelector
from config  import SEED, N_ARMS, ARM_NAMES, DEFAULT_C_UCB
from bandit_all import (
    WON_15, TANNER_20, TANNER_DAYS,
    COGBCI_SUBS, COGBCI_DAYS,
    CHO_20, CHO_DAYS, LEE_20, LEE_DAYS,
    make_splits, pool_enrol,
    DATASETS,
)

OUT_DIR  = "results_feature_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

P        = 10
N_ITER   = 1000
LAMBDA   = 0.3
FS       = 250.0
P_SWEEP  = list(range(2, 51, 2))


def riemannian_vec(trial: np.ndarray) -> np.ndarray:
    cov = riemann_covariance(trial)
    vec = cov[np.triu_indices_from(cov)]
    return vec / (np.linalg.norm(vec) + 1e-9)

def euclidean_vec(trial: np.ndarray) -> np.ndarray:
    cov = np.cov(trial) + np.eye(trial.shape[0]) * 1e-6
    vec = cov[np.triu_indices_from(cov)]
    return vec / (np.linalg.norm(vec) + 1e-9)

def psd_vec(trial: np.ndarray, fs: float = FS) -> np.ndarray:
    bands    = [(1, 4), (4, 8), (8, 13), (13, 30)]
    C, T     = trial.shape
    nperseg  = min(T, 256)
    feats    = []
    for ch in range(C):
        freqs, psd = welch(trial[ch], fs=fs, nperseg=nperseg)
        for flo, fhi in bands:
            mask = (freqs >= flo) & (freqs < fhi)
            feats.append(float(psd[mask].mean()) if mask.any() else 0.0)
    vec = np.array(feats)
    return vec / (np.linalg.norm(vec) + 1e-9)

FEATURE_FNS = {
    "riemannian": riemannian_vec,
    "euclidean":  euclidean_vec,
    "psd":        psd_vec,
}


def feat_array(trial_list, feat_fn):
    return np.array([feat_fn(x) for x in trial_list])

def build_templates(feat_enrol, y_enrol):
    templates = {}
    for cls in np.unique(y_enrol):
        tmpl = feat_enrol[y_enrol == cls].mean(axis=0)
        templates[int(cls)] = tmpl / (np.linalg.norm(tmpl) + 1e-9)
    return templates

def compute_tau(templates, feat_enrol, y_enrol, p):
    genuine = [cosine_sim(v, templates[int(c)])
               for c in np.unique(y_enrol) for v in feat_enrol[y_enrol == c]]
    return float(np.percentile(genuine, p))

def compute_eer(templates, feat_enrol, y_enrol, rng):
    """Sweep p to find EER."""
    best_diff, eer_p, eer_val = 1e9, 2, 0.5
    for p in P_SWEEP:
        tau = compute_tau(templates, feat_enrol, y_enrol, p)
        classes = list(templates.keys())
        frrs, fars = [], []
        for cls in classes:
            mask = (y_enrol == cls)
            if not mask.any():
                continue
            gen_scores = [cosine_sim(v, templates[cls]) for v in feat_enrol[mask]]
            frrs.append(float(np.mean(np.array(gen_scores) < tau)))
            others = [c for c in classes if c != cls]
            if others:
                ic = int(rng.choice(others))
                imp = feat_enrol[y_enrol == ic]
                if len(imp):
                    imp_scores = [cosine_sim(v, templates[cls]) for v in imp]
                    fars.append(float(np.mean(np.array(imp_scores) >= tau)))
        if not frrs or not fars:
            continue
        frr_m = np.mean(frrs)
        far_m = np.mean(fars)
        diff  = abs(frr_m - far_m)
        if diff < best_diff:
            best_diff = diff
            eer_p     = p
            eer_val   = (frr_m + far_m) / 2
    return eer_p, eer_val


def attack_loop(templates, tau, X_test, y_test, n_iter, rng, feat_fn, lam=LAMBDA):
    X_arr   = np.array(X_test)
    classes = np.array(list(templates.keys()))
    logs    = []

    ucb  = UCBSelector(n_arms=N_ARMS, c=DEFAULT_C_UCB)
    arms = AttackArms(rng=rng, s_freq=int(FS))

    for _ in range(n_iter):
        arm = ucb.select()
        vi  = int(rng.integers(len(X_arr)))
        X0  = X_arr[vi]
        ts  = int(y_test[vi])

        imp_trial = None
        if arm == 4:
            others = [c for c in classes if c != ts]
            if len(others):
                ic = int(rng.choice(others))
                im = np.where(y_test == ic)[0]
                if len(im):
                    imp_trial = X_arr[int(rng.choice(im))]
        try:
            Xp, _ = arms.apply(arm, X0, imp_trial)
        except Exception:
            ucb.update(arm, 0.0); continue

        others = [c for c in classes if c != ts]
        tgt    = int(rng.choice(others)) if len(others) else ts
        if tgt not in templates:
            ucb.update(arm, 0.0); continue

        q        = feat_fn(Xp)
        score    = cosine_sim(q, templates[tgt])
        accepted = score >= tau
        delta    = float(np.linalg.norm(Xp - X0) / (np.linalg.norm(X0) + 1e-9))
        reward   = (1 - lam) * float(accepted) + lam * max(0.0, 1.0 - delta)
        ucb.update(arm, reward)
        logs.append(bool(accepted))

    return float(np.mean(logs)) if logs else 0.0


def run_dataset(ds_name, ds_cfg, feat_name, feat_fn):
    sessions = load_sessions(ds_cfg["participants"], ds_cfg["days"], ds_cfg["erp_root"])
    if not sessions:
        print(f"  [{ds_name}/{feat_name}] no sessions found, skipping")
        return None

    rng              = np.random.default_rng(SEED)
    all_participants = sorted({s["participant"] for s in sessions})
    cls2idx          = {p: i for i, p in enumerate(all_participants)}
    splits           = make_splits(sessions, rng)
    by_day           = pool_enrol(splits)

    per_subj_adv, per_subj_base = [], []
    eer_vals = []
    first_day = True

    for day, day_splits in sorted(by_day.items()):
        all_enrol_X = [x for sp in day_splits for x in sp["X_enrol"]]
        all_enrol_y = [cls2idx[sp["participant"]] for sp in day_splits
                       for _ in sp["X_enrol"]]
        all_test_X  = [x for sp in day_splits for x in sp["X_test"]]
        all_test_y  = [cls2idx[sp["participant"]] for sp in day_splits
                       for _ in sp["X_test"]]
        y_enrol = np.array(all_enrol_y, dtype=int)
        y_test  = np.array(all_test_y,  dtype=int)

        feat_enrol = feat_array(all_enrol_X, feat_fn)
        templates  = build_templates(feat_enrol, y_enrol)
        tau        = compute_tau(templates, feat_enrol, y_enrol, P)

        # baseline FAR at p=10
        feat_test  = feat_array(all_test_X, feat_fn)
        classes    = list(templates.keys())
        for cls in classes:
            mask = (y_test == cls)
            if not mask.any():
                continue
            others = [c for c in classes if c != cls]
            if not others:
                continue
            ic  = int(rng.choice(others))
            imp = feat_test[y_test == ic]
            if len(imp):
                imp_scores = [cosine_sim(v, templates[cls]) for v in imp]
                per_subj_base.append(float(np.mean(np.array(imp_scores) >= tau)))

        if first_day:
            eer_p, eer_val = compute_eer(templates, feat_enrol, y_enrol, rng)
            eer_vals.append(eer_val)
            first_day = False

        # adversarial FAR
        torch_seed = SEED + P * 31 + abs(hash(day)) % 10_000
        rng_atk    = np.random.default_rng(torch_seed)
        adv_far    = attack_loop(templates, tau, all_test_X, y_test,
                                 N_ITER, rng_atk, feat_fn)
        per_subj_adv.append(adv_far)

    result = dict(
        dataset      = ds_name,
        feature      = feat_name,
        EER          = float(np.mean(eer_vals))   if eer_vals      else float("nan"),
        base_FAR_p10 = float(np.mean(per_subj_base)) if per_subj_base else float("nan"),
        adv_FAR_p10  = float(np.mean(per_subj_adv))  if per_subj_adv  else float("nan"),
    )
    print(f"  [{ds_name}/{feat_name}] EER={result['EER']:.3f}  "
          f"base_FAR={result['base_FAR_p10']:.3f}  adv_FAR={result['adv_FAR_p10']:.3f}")
    return result


rows = []
for ds_name, ds_cfg in DATASETS.items():
    print(f"\n=== {ds_name} ===")
    for feat_name, feat_fn in FEATURE_FNS.items():
        r = run_dataset(ds_name, ds_cfg, feat_name, feat_fn)
        if r:
            rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "feature_comparison.csv"), index=False)
print("\nSaved:", os.path.join(OUT_DIR, "feature_comparison.csv"))
print(df.to_string(index=False))


ds_order   = ["zhang", "won", "tanner_eo", "tanner_ec",
              "cogbci_eo", "cogbci_ec", "cho", "lee"]
feat_order = ["riemannian", "euclidean", "psd"]
colors     = {"riemannian": "steelblue", "euclidean": "darkorange", "psd": "seagreen"}

x   = np.arange(len(ds_order))
w   = 0.25
fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)

for ax, metric, title in zip(axes,
                              ["base_FAR_p10", "adv_FAR_p10"],
                              ["Baseline FAR ($p=10$)", "Adversarial FAR ($p=10$, UCB)"]):
    for i, feat in enumerate(feat_order):
        sub = df[df["feature"] == feat].set_index("dataset")
        vals = [sub.loc[ds, metric] if ds in sub.index else float("nan")
                for ds in ds_order]
        ax.bar(x + (i - 1) * w, vals, w, label=feat, color=colors[feat], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_order, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("FAR", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

plt.suptitle("Feature space comparison: Riemannian vs. Euclidean vs. PSD", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_comparison.png"), dpi=150)
plt.close()
print("Saved:", os.path.join(OUT_DIR, "feature_comparison.png"))
