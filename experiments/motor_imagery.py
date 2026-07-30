"""Run analysis for motor imagery datasets only (Cho + Lee)."""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import os, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from helpers import load_sessions
from utils import trial_to_cov_vec, cosine_sim
from arms import AttackArms
from ucb import UCBSelector
from config import SEED, N_ARMS, ARM_NAMES, DEFAULT_C_UCB
from bandit_all import (
    ENROL_FRAC, LAMBDA, P_SWEEP, P_ATTACK, N_ITER,
    build_templates, compute_tau, cov_array,
    make_splits, pool_enrol,
    threshold_sweep, run_ucb_attack,
)

_rng_cho = np.random.default_rng(42)
CHO_20 = sorted(_rng_cho.choice([f"s{i:02d}" for i in range(1, 53)], size=20, replace=False).tolist())
_rng_lee = np.random.default_rng(42)
LEE_20  = sorted(_rng_lee.choice([f"sub-{i:02d}" for i in range(1, 55)], size=20, replace=False).tolist())

print(f"Cho 20: {CHO_20}")
print(f"Lee 20: {LEE_20}\n")

DATASETS = {
    "cho": dict(
        erp_root    = "/Users/a1234/red_team_mc/data/preprocessed/cho",
        participants= CHO_20,
        days        = ["session1"],
        out_dir     = "results_cho",
    ),
    "lee": dict(
        erp_root    = "/Users/a1234/red_team_mc/data/preprocessed/lee",
        participants= LEE_20,
        days        = ["session1", "session2"],
        out_dir     = "results_lee",
    ),
}

summary = {}

for ds_name, cfg in DATASETS.items():
    print(f"\n{'='*64}\n  {ds_name.upper()}\n{'='*64}")
    os.makedirs(cfg["out_dir"], exist_ok=True)

    sessions = load_sessions(cfg["participants"], cfg["days"], erp_root=cfg["erp_root"])
    print(f"Sessions loaded: {len(sessions)}  "
          f"(subjects: {len({s['participant'] for s in sessions})})")

    sw, eer_p, eer = threshold_sweep(sessions, cfg["out_dir"], ds_name)
    adv            = run_ucb_attack(sessions,  cfg["out_dir"], ds_name)

    summary[ds_name] = dict(sweep=sw, eer_p=eer_p, eer=eer, adv=adv)

print("\n\n" + "="*64)
print("SUMMARY")
print("="*64)

for ds in ["cho", "lee"]:
    s   = summary[ds]
    p10 = s["sweep"][10]
    print(f"\n{ds.upper()}")
    print(f"  EER ≈ {s['eer']:.3f} at p={s['eer_p']}")
    print(f"  p=10: FRR={p10['mean_frr']:.3f}±{p10['std_frr']:.3f}  "
          f"FAR={p10['mean_far']:.3f}±{p10['std_far']:.3f}")
    for p in P_ATTACK:
        r = s["adv"][p]
        best = max(r["arm"], key=lambda n: r["arm"][n][0])
        print(f"  adv p={p:>2}: FAR={r['mean_far']:.3f}±{r['std_far']:.3f}  best={best}")

print("\nDone.")
