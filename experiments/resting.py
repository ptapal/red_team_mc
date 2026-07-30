"""Run analysis for resting-state datasets only (Tanner EO/EC + COG-BCI EO/EC)."""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import os, warnings
import matplotlib; matplotlib.use("Agg")

warnings.filterwarnings("ignore")

from helpers  import load_sessions
from datasets import get_datasets
from bandit_all import P_SWEEP, P_ATTACK, SEED, threshold_sweep, run_ucb_attack

_KEYS    = ("tanner_eo", "tanner_ec", "cogbci_eo", "cogbci_ec")
DATASETS = {k: v for k, v in get_datasets("results").items() if k in _KEYS}

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
print("SUMMARY  (resting state, 1–30 Hz)")
print("="*64)

for ds in ["tanner_eo", "tanner_ec", "cogbci_eo", "cogbci_ec"]:
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
