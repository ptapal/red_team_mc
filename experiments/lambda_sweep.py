"""
Lambda ablation: UCB bandit at p=10 across λ ∈ {0.0, 0.1, 0.3, 0.5}.

Tests whether the arm preference ordering is robust to the dual-reward
weight λ. Runs on Zhang (RSVP) and Wang-EO (resting-state) only to keep
compute tractable; these two cover the paradigms with clearest arm preference.

Output
------
  results_lambda_ablation/
    arm_means_zhang.csv      — per-arm mean reward at each λ, Zhang
    arm_means_tanner_eo.csv  — per-arm mean reward at each λ, Wang-EO
    arm_order_zhang.csv      — rank ordering of arms at each λ, Zhang
    arm_order_tanner_eo.csv  — rank ordering of arms at each λ, Wang-EO
    lambda_ablation.tex      — LaTeX table (best arm + FAR per λ)
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
from bandit_all import (
    TANNER_20, WON_15, TANNER_DAYS,
    make_splits, pool_enrol, build_templates, compute_tau, cov_array,
    _attack_loop,
    CHO_20, LEE_20, COGBCI_SUBS, CHO_DAYS, LEE_DAYS, COGBCI_DAYS,
)

OUT_DIR = "results_lambda_ablation"
os.makedirs(OUT_DIR, exist_ok=True)

P_TEST   = 10
N_ITER   = 1000
LAMBDAS  = [0.0, 0.1, 0.3, 0.5]

DATASETS = {
    "zhang": dict(
        erp_root    = "/Users/a1234/red_team_mc/data/preprocessed/zhang/won_channels",
        participants= [f"S{i}" for i in range(1, 16)],
        days        = ["Day_1", "Day_7", "Day_80", "Day_200"],
        label       = "Zhang (RSVP)",
    ),
    "won": dict(
        erp_root    = "/Users/a1234/red_team_mc/data/preprocessed/won",
        participants= WON_15,
        days        = ["Day_1"],
        label       = "Won (RSVP)",
    ),
    "tanner_eo": dict(
        erp_root    = "/Users/a1234/red_team_mc/data/preprocessed/wang/eo",
        participants= TANNER_20,
        days        = TANNER_DAYS,
        label       = "Wang-EO (resting)",
    ),
    "tanner_ec": dict(
        erp_root    = "/Users/a1234/red_team_mc/data/preprocessed/wang/ec",
        participants= TANNER_20,
        days        = TANNER_DAYS,
        label       = "Wang-EC (resting)",
    ),
    "cogbci_eo": dict(
        erp_root    = "/Users/a1234/red_team_mc/data/preprocessed/cogbci/eo",
        participants= COGBCI_SUBS,
        days        = COGBCI_DAYS,
        label       = "COG-BCI-EO (resting)",
    ),
    "cogbci_ec": dict(
        erp_root    = "/Users/a1234/red_team_mc/data/preprocessed/cogbci/ec",
        participants= COGBCI_SUBS,
        days        = COGBCI_DAYS,
        label       = "COG-BCI-EC (resting)",
    ),
    "cho": dict(
        erp_root    = "/Users/a1234/red_team_mc/data/preprocessed/cho",
        participants= CHO_20,
        days        = CHO_DAYS,
        label       = "Cho (motor imagery)",
    ),
    "lee": dict(
        erp_root    = "/Users/a1234/red_team_mc/data/preprocessed/lee",
        participants= LEE_20,
        days        = LEE_DAYS,
        label       = "Lee (motor imagery)",
    ),
}


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
    all_rows   = {}   # ds_name -> list of dicts (one per lam)
    tex_rows   = []

    for ds_name, cfg in DATASETS.items():
        print(f"\n{'='*60}\n  {cfg['label']}\n{'='*60}")
        sessions = load_sessions(cfg["participants"], cfg["days"],
                                 erp_root=cfg["erp_root"])
        if not sessions:
            print("  No sessions — skipping.")
            continue

        rows = []
        for lam in LAMBDAS:
            print(f"  λ={lam:.1f} ...", end=" ", flush=True)
            arm_means, far, best = run_dataset_at_lambda(sessions, lam)
            row = {"lambda": lam, "best_arm": best, "FAR": far}
            row.update({f"reward_{n}": arm_means[n] for n in ARM_NAMES})
            rows.append(row)
            print(f"best={best}  FAR={far:.3f}")

        all_rows[ds_name] = rows

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT_DIR, f"arm_means_{ds_name}.csv"), index=False)

        # rank ordering per lam
        rank_rows = []
        for r in rows:
            sorted_arms = sorted(ARM_NAMES,
                                 key=lambda n: r[f"reward_{n}"], reverse=True)
            rank_rows.append({"lambda": r["lambda"],
                              **{f"rank_{i+1}": a
                                 for i, a in enumerate(sorted_arms)}})
        pd.DataFrame(rank_rows).to_csv(
            os.path.join(OUT_DIR, f"arm_order_{ds_name}.csv"), index=False)

        for r in rows:
            tex_rows.append(dict(
                dataset = cfg["label"],
                lam     = r["lambda"],
                best    = r["best_arm"],
                far     = r["FAR"],
            ))

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{UCB arm-preference robustness to $\lambda$ at $p=10$.",
        r"Best arm and adversarial FAR on two representative datasets.",
        r"$\lambda=0$ is pure acceptance reward; $\lambda=0.3$ is the",
        r"setting used throughout the paper.}",
        r"\label{tab:lambda_ablation}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Dataset & $\lambda$ & Best arm & FAR \\",
        r"\midrule",
    ]
    current_ds = None
    for r in tex_rows:
        if r["dataset"] != current_ds:
            if current_ds is not None:
                lines.append(r"\addlinespace[3pt]")
            current_ds = r["dataset"]
        lam_str = r"$0$ (acceptance only)" if r["lam"] == 0.0 else f"${r['lam']}$"
        lines.append(
            f"{r['dataset']} & {lam_str} & {r['best']} & {r['far']:.3f} \\\\"
        )
        # only print dataset name on first row
        tex_rows_after = [x for x in tex_rows if x["dataset"] == r["dataset"]]
        # blank dataset col for subsequent rows
    # rewrite with blank dataset name for continuation rows
    lines_clean = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{UCB arm-preference robustness to $\lambda$ at $p=10$.",
        r"Best arm and adversarial FAR on two representative datasets.",
        r"$\lambda=0$ is pure acceptance reward; $\lambda=0.3$ is the",
        r"setting used throughout the paper.}",
        r"\label{tab:lambda_ablation}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Dataset & $\lambda$ & Best arm & FAR \\",
        r"\midrule",
    ]
    current_ds = None
    for r in tex_rows:
        ds_col = r["dataset"] if r["dataset"] != current_ds else ""
        current_ds = r["dataset"]
        lam_str = r"$0$ (acceptance only)" if r["lam"] == 0.0 else f"${r['lam']}$"
        lines_clean.append(
            f"{ds_col} & {lam_str} & {r['best']} & {r['far']:.3f} \\\\"
        )
    lines_clean += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex = "\n".join(lines_clean)
    with open(os.path.join(OUT_DIR, "lambda_ablation.tex"), "w") as f:
        f.write(tex)

    print("\n--- LaTeX table ---")
    print(tex)
    print(f"\nSaved to {OUT_DIR}/")
