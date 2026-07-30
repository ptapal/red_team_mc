"""
Wilcoxon / Mann-Whitney U test: best vs. worst UCB arm per dataset at p=10.

Re-runs the UCB attack loop at p=10 for all 8 datasets (no neural networks,
fast) to recover per-iteration reward sequences.  Then tests whether the best
and worst arm produce statistically different reward distributions using the
one-sided Mann-Whitney U test (equivalent to Wilcoxon rank-sum; more
appropriate than signed-rank because the two arm reward sequences have
unequal length).

Outputs
-------
wilcoxon_results.csv   — per-dataset U statistic, p-value, arm names
wilcoxon_table.tex     — LaTeX table ready to paste
"""

import os, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

from helpers import load_sessions
from config  import SEED
from run_analysis_normalised import (
    WON_15, TANNER_20, CHO_20, LEE_20,
    TANNER_DAYS, COGBCI_DAYS, COGBCI_SUBS, CHO_DAYS, LEE_DAYS,
    make_splits, pool_enrol, build_templates, compute_tau, cov_array,
    _attack_loop, ARM_NAMES, LAMBDA,
)

P_TEST = 10

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

SHORT_NAMES = {
    "Gaussian Noise":     "Gauss.\ noise",
    "Correlated Noise":   "Corr.\ noise",
    "Temporal Distortion":"Temporal dist.",
    "Freq Manipulation":  "Freq.\ manip.",
    "Cov Interpolation":  "Cov.\ interp.",
    "Replay + Jitter":    "Replay+jitter",
    "Channel Coupling":   "Chan.\ coupling",
    "Eigenvalue Perturb": "Eigenval.\ pert.",
}


def run_dataset(sessions, ds_name):
    rng = np.random.default_rng(SEED)
    all_participants = sorted({s["participant"] for s in sessions})
    cls2idx = {p: i for i, p in enumerate(all_participants)}
    splits  = make_splits(sessions, rng)
    by_day  = pool_enrol(splits)

    arm_rewards = {name: [] for name in ARM_NAMES}

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

        logs, _, _ = _attack_loop(templates, tau, all_test_X, y_test,
                                  1000, rng)
        for l in logs:
            arm_rewards[l["arm_name"]].append(l["reward"])

    arm_means = {n: (np.mean(v) if v else 0.0) for n, v in arm_rewards.items()}
    best_arm  = max(arm_means, key=arm_means.get)
    worst_arm = min(arm_means, key=arm_means.get)

    rb = np.array(arm_rewards[best_arm])
    rw = np.array(arm_rewards[worst_arm])

    if len(rb) < 2 or len(rw) < 2 or arm_means[best_arm] == arm_means[worst_arm]:
        return dict(ds=ds_name, best_arm=best_arm, worst_arm=worst_arm,
                    best_mean=arm_means[best_arm], worst_mean=arm_means[worst_arm],
                    U=float("nan"), p_value=float("nan"),
                    n_best=len(rb), n_worst=len(rw), sig="—")

    U, pval = stats.mannwhitneyu(rb, rw, alternative="greater")
    sig = ("$p<0.001$" if pval < 0.001 else
           f"$p={pval:.3f}$" if pval < 0.05 else
           f"$p={pval:.3f}$\,(n.s.)")
    return dict(ds=ds_name, best_arm=best_arm, worst_arm=worst_arm,
                best_mean=arm_means[best_arm], worst_mean=arm_means[worst_arm],
                U=float(U), p_value=float(pval),
                n_best=int(len(rb)), n_worst=int(len(rw)), sig=sig)


if __name__ == "__main__":
    rows = []
    for ds_name, cfg in DATASETS.items():
        print(f"\n{'='*60}\n  {ds_name.upper()}\n{'='*60}")
        sessions = load_sessions(cfg["participants"], cfg["days"],
                                 erp_root=cfg["erp_root"])
        if not sessions:
            print("  No sessions — skipping.")
            continue
        print(f"  {len(sessions)} sessions, "
              f"{len({s['participant'] for s in sessions})} subjects")
        res = run_dataset(sessions, ds_name)
        res["label"] = cfg["label"]
        rows.append(res)
        print(f"  Best:  {res['best_arm']} (mean reward {res['best_mean']:.3f}, "
              f"n={res['n_best']})")
        print(f"  Worst: {res['worst_arm']} (mean reward {res['worst_mean']:.3f}, "
              f"n={res['n_worst']})")
        print(f"  Mann-Whitney U={res['U']:.1f},  {res['sig']}")

    df = pd.DataFrame(rows)
    df.to_csv("wilcoxon_results.csv", index=False)
    print("\nSaved wilcoxon_results.csv")

    # LaTeX table
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Mann-Whitney U test comparing best vs.\ worst UCB arm at $p=10$.",
        r"$H_0$: reward distributions of best and worst arm are identical.",
        r"$n_b$/$n_w$: number of pulls for best/worst arm (pooled across days).}",
        r"\label{tab:wilcoxon}",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Dataset & Best arm & Worst arm & $\bar{r}_b$ & $\bar{r}_w$ & Test \\",
        r"\midrule",
    ]
    paradigm_groups = [
        (["zhang","won"],             "RSVP"),
        (["tanner_eo","tanner_ec",
          "cogbci_eo","cogbci_ec"],   "Resting-state"),
        (["cho","lee"],               "Motor imagery"),
    ]
    for ds_keys, group in paradigm_groups:
        group_rows = [r for r in rows if r["ds"] in ds_keys]
        if group_rows:
            lines.append(r"\addlinespace[3pt]")
            lines.append(r"\multicolumn{6}{l}{\textit{" + group + r"}} \\")
        for r in group_rows:
            b = SHORT_NAMES.get(r["best_arm"],  r["best_arm"])
            w = SHORT_NAMES.get(r["worst_arm"], r["worst_arm"])
            lines.append(
                f"{r['label']} & {b} & {w} & "
                f"{r['best_mean']:.3f} & {r['worst_mean']:.3f} & {r['sig']} \\\\"
            )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex = "\n".join(lines)
    with open("wilcoxon_table.tex", "w") as f:
        f.write(tex)
    print("Saved wilcoxon_table.tex")
    print("\n--- LaTeX table ---")
    print(tex)
