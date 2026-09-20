"""
Surrogate-gradient GAN (SGAN) against the Gaussian Naive Bayes authenticator.

Usage:  python experiments/run_sgan_gnb.py [--datasets a,b]
Writes results/sgan_gnb/<dataset>/adv_far_sgan.csv and results/sgan_gnb/summary.csv.
"""

import os
import csv
import time
import argparse
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

from helpers  import load_sessions
from config   import SEED
from utils    import trial_to_cov_vec
from datasets import get_datasets
from experiments.bandit_all import make_splits, pool_enrol, cov_array, P_ATTACK
from attacks.sgan import run_surrogate_gan_attack
from experiments.run_vae_clf import stable_seed
from experiments.run_gnb import build_gnb_models, gnb_score, compute_tau_gnb

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
LAMBDA   = 0.3
N_ITER   = 1000
OUT_ROOT = "results/sgan_gnb"
_EPS     = 1e-12


def _gnb_quadratic_params(clf):
    """(a, b, c) such that log-odds(q) = a.q^2 + b.q + c, for one-vs-all GNB."""
    classes = list(clf.classes_)
    if 1 not in classes:
        return None
    pos, neg = classes.index(1), classes.index(0) if 0 in classes else None
    if neg is None:
        return None
    mu1, mu0 = clf.theta_[pos], clf.theta_[neg]
    v1 = np.maximum(clf.var_[pos], _EPS)
    v0 = np.maximum(clf.var_[neg], _EPS)

    a = 0.5 * (1.0 / v0 - 1.0 / v1)
    b = mu1 / v1 - mu0 / v0
    c = (0.5 * np.sum(mu0 ** 2 / v0 - mu1 ** 2 / v1)
         + 0.5 * np.sum(np.log(v0) - np.log(v1))
         + np.log(max(clf.class_prior_[pos], _EPS))
         - np.log(max(clf.class_prior_[neg], _EPS)))
    return a, b, float(c)


def gnb_conditioning(models: dict) -> dict:
    """{class: concat(a_k, b_k)} -- the identity-specific decision parameters."""
    cond = {}
    for cls, clf in models.items():
        parms = _gnb_quadratic_params(clf)
        if parms is None:
            continue
        a, b, _ = parms
        cond[int(cls)] = np.concatenate([a, b]).astype(np.float64)
    return cond


def gnb_logodds(models: dict, cls: int, q: np.ndarray) -> float:
    """Monotone reparameterisation of gnb_score: same ranking, unbounded range."""
    parms = _gnb_quadratic_params(models[cls])
    if parms is None:
        return 0.0
    a, b, c = parms
    return float(np.dot(a, q ** 2) + np.dot(b, q) + c)


def run_dataset(sessions, ds_name):
    rng = np.random.default_rng(SEED)
    all_participants = sorted({s["participant"] for s in sessions})
    cls2idx = {p: i for i, p in enumerate(all_participants)}
    by_day  = pool_enrol(make_splits(sessions, rng))

    per_day = {}
    for day, day_splits in sorted(by_day.items()):
        enrol_X = [x for sp in day_splits for x in sp["X_enrol"]]
        enrol_y = np.array([cls2idx[sp["participant"]] for sp in day_splits
                            for _ in sp["X_enrol"]], dtype=int)
        cov_enrol = cov_array(enrol_X)
        models = build_gnb_models(cov_enrol, enrol_y)
        cond   = gnb_conditioning(models)
        if not cond:
            continue
        models = {k: v for k, v in models.items() if k in cond}
        per_day[day] = dict(
            models    = models,
            cond      = cond,
            cov_enrol = cov_enrol,
            y_enrol   = enrol_y,
            X_enrol   = enrol_X,
            X_test    = [x for sp in day_splits for x in sp["X_test"]],
            y_test    = np.array([cls2idx[sp["participant"]] for sp in day_splits
                                  for _ in sp["X_test"]], dtype=int),
        )

    results = {}
    for p in P_ATTACK:
        per_subj_far, budget_accepted = [], []
        zero_effort, genuine_rejected, fidelity_r = [], [], []

        for day, dc in per_day.items():
            models = dc["models"]
            tau = compute_tau_gnb(models, dc["cov_enrol"], dc["y_enrol"], p)
            score_fn     = lambda q, tgt: gnb_score(models[tgt], q)
            surrogate_fn = lambda q, tgt: gnb_logodds(models, tgt, q)

            logs, per_tgt, D, S, G = run_surrogate_gan_attack(
                dc["cond"], tau, dc["X_test"], dc["y_test"],
                dc["X_enrol"], dc["y_enrol"],
                N_ITER, rng, lam=LAMBDA, device=DEVICE,
                seed=stable_seed(day, p),
                score_fn=score_fn, surrogate_score_fn=surrogate_fn,
            )
            fid = getattr(S, "fidelity", None)
            if fid and not np.isnan(fid["r"]):
                fidelity_r.append(fid["r"])
            if not logs:
                continue
            for accepted_list in per_tgt.values():
                if accepted_list:
                    per_subj_far.append(float(np.mean(accepted_list)))
            budget_accepted += [l["budget"] for l in logs if l["accepted"]]
            zero_effort += [l["accepted_zero_effort"] for l in logs
                            if l["accepted_zero_effort"] is not None]

            for vec, cls in zip((trial_to_cov_vec(x) for x in dc["X_test"]),
                                dc["y_test"]):
                if int(cls) in models:
                    genuine_rejected.append(
                        float(gnb_score(models[int(cls)], vec)) < tau)

        mean_far = float(np.mean(per_subj_far))     if per_subj_far     else 0.0
        std_far  = float(np.std(per_subj_far))      if per_subj_far     else 0.0
        budget   = float(np.mean(budget_accepted))  if budget_accepted  else float("nan")
        base_far = float(np.mean(zero_effort))      if zero_effort      else float("nan")
        frr      = float(np.mean(genuine_rejected)) if genuine_rejected else float("nan")
        s_r      = float(np.mean(fidelity_r))       if fidelity_r       else float("nan")
        results[p] = (mean_far, std_far, budget, base_far,
                      mean_far - base_far, frr, s_r)
        print(f"  p={p:>2}: base {base_far:.3f} -> SGAN {mean_far:.3f} +- {std_far:.3f}"
              f"   lift {mean_far - base_far:+.3f}   FRR {frr:.3f}"
              f"   surrogate r {s_r:.3f}   budget {budget:.3f}", flush=True)

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="")
    args = ap.parse_args()

    datasets = get_datasets("results")
    if args.datasets:
        keep = {d.strip() for d in args.datasets.split(",")}
        datasets = {k: v for k, v in datasets.items() if k in keep}

    print(f"Device: {DEVICE} | datasets: {list(datasets)}")
    summary_rows = []

    for ds_name, cfg in datasets.items():
        sessions = load_sessions(cfg["participants"], cfg["days"],
                                 erp_root=cfg["erp_root"])
        if not sessions:
            print(f"{ds_name}: no sessions - skipping.")
            continue
        print(f"\n{'='*64}\n  {ds_name.upper()}  (SGAN vs GNB)\n{'='*64}", flush=True)
        t0 = time.time()
        res = run_dataset(sessions, ds_name)
        print(f"  elapsed: {time.time()-t0:.1f}s", flush=True)

        out_dir = os.path.join(OUT_ROOT, ds_name)
        os.makedirs(out_dir, exist_ok=True)
        cols = ["p", "base_FAR", "SGAN_FAR_mean", "SGAN_FAR_std", "lift",
                "genuine_FRR", "surrogate_r", "Budget_mean"]
        with open(os.path.join(out_dir, "adv_far_sgan.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for p in P_ATTACK:
                m, s, b, base, lift, frr, r = res[p]
                w.writerow([p, round(base, 4), round(m, 4), round(s, 4),
                            round(lift, 4), round(frr, 4), round(r, 4), round(b, 4)])
        for p in P_ATTACK:
            m, s, b, base, lift, frr, r = res[p]
            summary_rows.append(dict(dataset=ds_name, p=p,
                                     base_FAR=round(base, 4),
                                     SGAN_FAR_mean=round(m, 4),
                                     SGAN_FAR_std=round(s, 4),
                                     lift=round(lift, 4),
                                     genuine_FRR=round(frr, 4),
                                     surrogate_r=round(r, 4),
                                     Budget_mean=round(b, 4)))

    if summary_rows:
        os.makedirs(OUT_ROOT, exist_ok=True)
        out = os.path.join(OUT_ROOT, "summary.csv")
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nWrote {out}")
