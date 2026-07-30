"""
Threshold sweep: evaluate FRR and FAR across percentile values p
using the RiemannAuthenticator on the existing within-session splits.
Produces a plot of FRR and FAR vs p, and prints the results table.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import os

from helpers  import load_sessions, build_within_session_splits
from utils    import trial_to_cov_vec, cosine_sim
from config   import SEED, ENROL_FRAC
from datasets import get_datasets

_zhang = get_datasets()["zhang"]

print("Loading sessions …")
sessions = load_sessions(_zhang["participants"], _zhang["days"], erp_root=_zhang["erp_root"])
print(f"Found {len(sessions)} valid sessions.")

all_participants = sorted({s["participant"] for s in sessions})
cls2idx          = {p: i for i, p in enumerate(all_participants)}

rng      = np.random.default_rng(SEED)
day_data = build_within_session_splits(sessions, cls2idx, rng, ENROL_FRAC)

percentiles = list(range(2, 51, 2))   # 2, 4, 6, … 50

mean_frr, mean_far = [], []

for p in percentiles:
    frrs, fars = [], []

    for day, data in day_data.items():
        # compute enrolment covariance vectors
        cov_enrol = np.array([trial_to_cov_vec(x) for x in data["X_enrol"]])
        y_enrol   = data["y_enrol"]
        cov_test  = np.array([trial_to_cov_vec(x) for x in data["X_test"]])
        y_test    = data["y_test"]

        # build templates
        templates = {}
        for cls in np.unique(y_enrol):
            tmpl = cov_enrol[y_enrol == cls].mean(axis=0)
            templates[int(cls)] = tmpl / (np.linalg.norm(tmpl) + 1e-9)

        # calibrate tau at this percentile
        genuine_scores = []
        for cls in np.unique(y_enrol):
            tmpl = templates[int(cls)]
            for v in cov_enrol[y_enrol == cls]:
                genuine_scores.append(cosine_sim(v, tmpl))
        tau = float(np.percentile(genuine_scores, p))

        # evaluate on test set
        n_genuine  = 0; n_fr = 0
        n_impostor = 0; n_fa = 0

        classes = list(templates.keys())

        for i, (vec, true_cls) in enumerate(zip(cov_test, y_test)):
            true_cls = int(true_cls)
            if true_cls not in templates:
                continue

            # genuine attempt: claim own identity
            score   = cosine_sim(vec, templates[true_cls])
            n_genuine += 1
            if score < tau:
                n_fr += 1

            # impostor attempt: claim a random other identity
            other = [c for c in classes if c != true_cls]
            if not other:
                continue
            imp_cls = int(rng.choice(other))
            score_imp = cosine_sim(vec, templates[imp_cls])
            n_impostor += 1
            if score_imp >= tau:
                n_fa += 1

        if n_genuine > 0:
            frrs.append(n_fr / n_genuine)
        if n_impostor > 0:
            fars.append(n_fa / n_impostor)

    mean_frr.append(np.mean(frrs) if frrs else np.nan)
    mean_far.append(np.mean(fars)  if fars  else np.nan)

print(f"\n{'p':>4}  {'FRR':>8}  {'FAR':>8}")
print("-" * 24)
for p, frr, far in zip(percentiles, mean_frr, mean_far):
    marker = "  <-- p=10" if p == 10 else ""
    print(f"{p:>4}  {frr:>8.4f}  {far:>8.4f}{marker}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(percentiles, mean_frr, "o-", label="FRR (False Rejection Rate)", color="steelblue")
ax.plot(percentiles, mean_far, "s-", label="FAR (False Acceptance Rate)", color="tomato")
ax.axvline(10, color="gray", linestyle="--", linewidth=1.2, label="p = 10")
ax.set_xlabel("Threshold percentile $p$", fontsize=13)
ax.set_ylabel("Rate", fontsize=13)
ax.set_title("FRR and FAR vs. threshold percentile\n(averaged across days and subjects)", fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = "threshold_sweep.png"
plt.savefig(out, dpi=150)
print(f"\nPlot saved to {out}")
plt.show()
