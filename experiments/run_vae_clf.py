import os
import csv
import time
import zlib
import argparse
import warnings

import numpy as np
import torch

from utils import trial_to_cov_vec


def stable_seed(day, p: int) -> int:
    return (SEED + p * 31 + zlib.crc32(str(day).encode()) % 10_000) % (2**31 - 1)

warnings.filterwarnings("ignore")

from helpers  import load_sessions
from config   import SEED
from datasets import get_datasets
from experiments.bandit_all import make_splits, pool_enrol, cov_array, P_ATTACK
from attacks.vae import run_vae_attack

from experiments.run_gnb import build_gnb_models, gnb_score, compute_tau_gnb
from experiments.run_svm import (build_svm_models, svm_score, oof_genuine_scores,
                     compute_tau_svm)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
LAMBDA   = 0.3
N_ITER   = 1000
OUT_ROOT = "results/vae_clf"


def _fit_gnb(cov_enrol, y_enrol):
    """Returns (models, tau_fn) where tau_fn(p) -> threshold."""
    models = build_gnb_models(cov_enrol, y_enrol)
    return models, (lambda p: compute_tau_gnb(models, cov_enrol, y_enrol, p))


def _fit_svm(cov_enrol, y_enrol):
    models = build_svm_models(cov_enrol, y_enrol)
    oof    = oof_genuine_scores(cov_enrol, y_enrol)
    return models, (lambda p: compute_tau_svm(oof, p))


CLASSIFIERS = {
    "gnb": dict(fit=_fit_gnb, score=gnb_score),
    "svm": dict(fit=_fit_svm, score=svm_score),
}


def run_dataset(sessions, clf_name, ds_name):
    """Returns {p: (mean_far, std_far, mean_budget)}."""
    spec = CLASSIFIERS[clf_name]
    rng  = np.random.default_rng(SEED)

    all_participants = sorted({s["participant"] for s in sessions})
    cls2idx = {p: i for i, p in enumerate(all_participants)}
    by_day  = pool_enrol(make_splits(sessions, rng))

    # Fit once per day and reuse across operating points: the model does not
    # depend on p, only tau does.
    per_day = {}
    for day, day_splits in sorted(by_day.items()):
        enrol_X = [x for sp in day_splits for x in sp["X_enrol"]]
        enrol_y = [cls2idx[sp["participant"]] for sp in day_splits
                   for _ in sp["X_enrol"]]
        cov_enrol = cov_array(enrol_X)
        y_enrol   = np.array(enrol_y, dtype=int)
        models, tau_fn = spec["fit"](cov_enrol, y_enrol)
        if not models:
            continue
        per_day[day] = dict(
            models  = models,
            tau_fn  = tau_fn,
            X_test  = [x for sp in day_splits for x in sp["X_test"]],
            y_test  = np.array([cls2idx[sp["participant"]] for sp in day_splits
                                for _ in sp["X_test"]], dtype=int),
        )

    results = {}
    for p in P_ATTACK:
        per_subj_far, budget_accepted = [], []
        zero_effort, genuine_rejected = [], []

        for day, dc in per_day.items():
            tau = dc["tau_fn"](p)
            models = dc["models"]
            score_fn = lambda q, tgt: spec["score"](models[tgt], q)

            logs, per_tgt = run_vae_attack(
                models, tau, dc["X_test"], dc["y_test"],
                N_ITER, rng, lam=LAMBDA, device=DEVICE,
                seed=stable_seed(day, p), score_fn=score_fn,
            )
            if not logs:
                continue
            for accepted_list in per_tgt.values():
                if accepted_list:
                    per_subj_far.append(float(np.mean(accepted_list)))
            budget_accepted += [l["budget"] for l in logs if l["accepted"]]
            zero_effort += [l["accepted_zero_effort"] for l in logs
                            if l["accepted_zero_effort"] is not None]

            # Held-out genuine FRR at this tau: each test trial scored against
            # its own identity. Reported alongside FAR because a calibration
            # that misses its nominal FRR target makes FAR uninterpretable
            # (cf. GNB on Zhang/Won, main text 3.3).
            for vec, cls in zip((trial_to_cov_vec(x) for x in dc["X_test"]),
                                dc["y_test"]):
                if int(cls) in models:
                    genuine_rejected.append(
                        float(spec["score"](models[int(cls)], vec)) < tau)

        mean_far = float(np.mean(per_subj_far))    if per_subj_far    else 0.0
        std_far  = float(np.std(per_subj_far))     if per_subj_far    else 0.0
        budget   = float(np.mean(budget_accepted)) if budget_accepted else float("nan")
        base_far = float(np.mean(zero_effort))     if zero_effort     else float("nan")
        frr      = float(np.mean(genuine_rejected)) if genuine_rejected else float("nan")
        results[p] = (mean_far, std_far, budget, base_far, mean_far - base_far, frr)
        print(f"  [{clf_name}] p={p:>2}: base {base_far:.3f} -> VAE {mean_far:.3f}"
              f" +- {std_far:.3f}   lift {mean_far - base_far:+.3f}"
              f"   FRR {frr:.3f}   budget {budget:.3f}", flush=True)

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf", default="both", choices=["gnb", "svm", "both"])
    ap.add_argument("--datasets", default="", help="comma-separated subset")
    args = ap.parse_args()

    datasets = get_datasets("results")
    if args.datasets:
        keep = {d.strip() for d in args.datasets.split(",")}
        datasets = {k: v for k, v in datasets.items() if k in keep}
    clfs = ["gnb", "svm"] if args.clf == "both" else [args.clf]

    print(f"Device: {DEVICE} | classifiers: {clfs} | datasets: {list(datasets)}")
    summary_rows = []

    for ds_name, cfg in datasets.items():
        sessions = load_sessions(cfg["participants"], cfg["days"],
                                 erp_root=cfg["erp_root"])
        if not sessions:
            print(f"{ds_name}: no sessions found - skipping.")
            continue
        for clf_name in clfs:
            print(f"\n{'='*64}\n  {ds_name.upper()}  (VAE vs {clf_name.upper()})"
                  f"\n{'='*64}", flush=True)
            t0 = time.time()
            res = run_dataset(sessions, clf_name, ds_name)
            print(f"  elapsed: {time.time()-t0:.1f}s", flush=True)

            out_dir = os.path.join(OUT_ROOT, clf_name, ds_name)
            os.makedirs(out_dir, exist_ok=True)
            cols = ["p", "base_FAR", "VAE_FAR_mean", "VAE_FAR_std", "lift",
                    "genuine_FRR", "Budget_mean"]
            with open(os.path.join(out_dir, "adv_far_vae.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(cols)
                for p in P_ATTACK:
                    m, s, b, base, lift, frr = res[p]
                    w.writerow([p, round(base, 4), round(m, 4), round(s, 4),
                                round(lift, 4), round(frr, 4), round(b, 4)])
            for p in P_ATTACK:
                m, s, b, base, lift, frr = res[p]
                summary_rows.append(dict(dataset=ds_name, classifier=clf_name,
                                         p=p, base_FAR=round(base, 4),
                                         VAE_FAR_mean=round(m, 4),
                                         VAE_FAR_std=round(s, 4),
                                         lift=round(lift, 4),
                                         genuine_FRR=round(frr, 4),
                                         Budget_mean=round(b, 4)))

    if summary_rows:
        os.makedirs(OUT_ROOT, exist_ok=True)
        out = os.path.join(OUT_ROOT, "summary.csv")
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nWrote {out}")
