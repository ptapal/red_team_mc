"""
Lambda ablation: UCB bandit at p=10 across lam in {0.0, 0.1, 0.3, 0.5}.

Tests whether the arm preference ordering is robust to the dual-reward weight lambda.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from helpers import load_sessions
from config  import SEED, ARM_NAMES
from datasets import get_datasets
from bandit_all import (
    make_splits, pool_enrol, build_templates, compute_tau, cov_array,
    _attack_loop,
)

OUT_DIR = "results_lambda_ablation"
os.makedirs(OUT_DIR, exist_ok=True)

P_TEST   = 10
N_ITER   = 1000
LAMBDAS  = [0.0, 0.1, 0.3, 0.5]

DATASETS = get_datasets("results_lambda_ablation")


def run_dataset_at_lambda(sessions, lam):
    rng = np.random.default_rng(SEED)
    all_participants = sorted({s["participant"] for s in sessions})
    cls2idx = {p: i for i, p in enumerate(all_participants)}
    splits  = make_splits(sessions, rng)
    by_day  = pool_enrol(splits)

    arm_rewards  = {name: [] for name in ARM_NAMES}
    per_subj_far = []

    for day, day_splits in sorted(by_day.items()):
        all_enrol_X = [x for sp in day_splits for x in sp["X_enrol"]]
        all_enrol_y = [cls2idx[sp["participant"]] for sp in day_splits
                       for _ in sp["X_enrol"]]
        cov_enrol = cov_array(all_enrol_X)
        y_enrol   = np.array(all_enrol_y, dtype=int)
        templates = build_templates(cov_enrol, y_enrol)
        tau       = compute_tau(templates, cov_enrol, y_enrol, P_TEST)

        all_test_X = [x for sp in day_splits for x in sp["X_test"]]
        all_test_y = [cls2idx[sp["participant"]] for sp in day_splits
                      for _ in sp["X_test"]]
        y_test = np.array(all_test_y, dtype=int)

        logs, per_tgt, _ = _attack_loop(templates, tau, all_test_X, y_test,
                                        N_ITER, rng, lam=lam)
        for l in logs:
            arm_rewards[l["arm_name"]].append(l["reward"])

        for cls_idx, accepted_list in per_tgt.items():
            if accepted_list:
                per_subj_far.append(float(np.mean(accepted_list)))

    arm_means = {n: (float(np.mean(v)) if v else 0.0)
                 for n, v in arm_rewards.items()}
    far_mean  = float(np.mean(per_subj_far)) if per_subj_far else 0.0
    best_arm  = max(arm_means, key=arm_means.get)

    return arm_means, far_mean, best_arm


if __name__ == "__main__":
    for ds_name, cfg in DATASETS.items():
        print(f"\n{'='*60}\n  {cfg['label']}\n{'='*60}")
        sessions = load_sessions(cfg["participants"], cfg["days"],
                                 erp_root=cfg["erp_root"])
        if not sessions:
            print("  No sessions -- skipping.")
            continue

        rows = []
        for lam in LAMBDAS:
            print(f"  lam={lam:.1f} ...", end=" ", flush=True)
            arm_means, far, best = run_dataset_at_lambda(sessions, lam)
            row = {"lambda": lam, "best_arm": best, "FAR": far}
            row.update({f"reward_{n}": arm_means[n] for n in ARM_NAMES})
            rows.append(row)
            print(f"best={best}  FAR={far:.3f}")

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT_DIR, f"arm_means_{ds_name}.csv"), index=False)

        rank_rows = []
        for r in rows:
            sorted_arms = sorted(ARM_NAMES,
                                 key=lambda n: r[f"reward_{n}"], reverse=True)
            rank_rows.append({"lambda": r["lambda"],
                              **{f"rank_{i+1}": a
                                 for i, a in enumerate(sorted_arms)}})
        pd.DataFrame(rank_rows).to_csv(
            os.path.join(OUT_DIR, f"arm_order_{ds_name}.csv"), index=False)

    print(f"\nSaved to {OUT_DIR}/")
