"""
VAE-based generative perturbation attack — all 8 datasets.

Exact same experimental conditions as the UCB MAB evaluation in
run_analysis_normalised.py:
  - Same subject subsamples, days, enrollment/test splits (SEED=42, rho=0.5)
  - Same reward function: r = (1-lam)*accepted + lam*max(0, 1-budget), lam=0.3
  - Same operating points: p in {5, 10, 15, 20, 25}
  - Same T=1000 iterations per (day, p) combination

Output per dataset
------------------
  results_vae_{ds}/adv_far_vae.csv      — FAR mean±std + mean budget, same format
                                           as adv_far_std.csv for direct comparison
  results_vae_{ds}/convergence_p10.png  — rolling FAR (window=100) vs. iteration

Comparison output
-----------------
  results_vae_comparison/vae_vs_ucb_p10.png  — side-by-side bar chart
  results_vae_comparison/vae_vs_ucb_full.csv — all datasets, all p values
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os
import warnings
import math

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

warnings.filterwarnings("ignore")

from helpers  import load_sessions
from config   import SEED
from utils    import trial_to_cov_vec, cosine_sim
from bandit_all import (
    WON_15, TANNER_20, CHO_20, LEE_20,
    TANNER_DAYS, COGBCI_DAYS, COGBCI_SUBS, CHO_DAYS, LEE_DAYS,
    DATASETS as UCB_DATASETS,
    make_splits, pool_enrol, build_templates, compute_tau, cov_array,
    P_ATTACK,
)
from attacks.vae import run_vae_attack

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAMBDA = 0.3
N_ITER = 1000

DATASETS = {
    "zhang": dict(
        erp_root     = "/Users/a1234/red_team_mc/data/preprocessed/zhang/won_channels",
        participants = [f"S{i}" for i in range(1, 16)],
        days         = ["Day_1", "Day_7", "Day_80", "Day_200"],
        out_dir      = "results_vae_zhang",
    ),
    "won": dict(
        erp_root     = "/Users/a1234/red_team_mc/data/preprocessed/won",
        participants = WON_15,
        days         = ["Day_1"],
        out_dir      = "results_vae_won",
    ),
    "tanner_eo": dict(
        erp_root     = "/Users/a1234/red_team_mc/data/preprocessed/wang/eo",
        participants = TANNER_20,
        days         = TANNER_DAYS,
        out_dir      = "results_vae_tanner_eo",
    ),
    "tanner_ec": dict(
        erp_root     = "/Users/a1234/red_team_mc/data/preprocessed/wang/ec",
        participants = TANNER_20,
        days         = TANNER_DAYS,
        out_dir      = "results_vae_tanner_ec",
    ),
    "cogbci_eo": dict(
        erp_root     = "/Users/a1234/red_team_mc/data/preprocessed/cogbci/eo",
        participants = COGBCI_SUBS,
        days         = COGBCI_DAYS,
        out_dir      = "results_vae_cogbci_eo",
    ),
    "cogbci_ec": dict(
        erp_root     = "/Users/a1234/red_team_mc/data/preprocessed/cogbci/ec",
        participants = COGBCI_SUBS,
        days         = COGBCI_DAYS,
        out_dir      = "results_vae_cogbci_ec",
    ),
    "cho": dict(
        erp_root     = "/Users/a1234/red_team_mc/data/preprocessed/cho",
        participants = CHO_20,
        days         = CHO_DAYS,
        out_dir      = "results_vae_cho",
    ),
    "lee": dict(
        erp_root     = "/Users/a1234/red_team_mc/data/preprocessed/lee",
        participants = LEE_20,
        days         = LEE_DAYS,
        out_dir      = "results_vae_lee",
    ),
}


# ── per-dataset runner ────────────────────────────────────────────────────────

def run_vae_dataset(sessions: list, out_dir: str, ds_name: str) -> dict:
    """
    Runs the VAE attack at all P_ATTACK operating points.

    Enrollment/test splits are drawn with the same seed and fraction (rho=0.5)
    as in run_ucb_attack(), so the two evaluations share identical splits.

    Returns {p: {mean_far, std_far, mean_budget, rolling_far (only at p=10)}}
    """
    rng = np.random.default_rng(SEED)
    all_participants = sorted({s["participant"] for s in sessions})
    cls2idx  = {p: i for i, p in enumerate(all_participants)}
    splits   = make_splits(sessions, rng)
    by_day   = pool_enrol(splits)
    all_results = {}

    for p in P_ATTACK:
        per_subj_far    = []
        budget_accepted = []
        rolling_far     = []      # only collected at p == 10

        for day, day_splits in sorted(by_day.items()):
            all_enrol_X = [x for sp in day_splits for x in sp["X_enrol"]]
            all_enrol_y = [cls2idx[sp["participant"]] for sp in day_splits
                           for _ in sp["X_enrol"]]
            cov_enrol = cov_array(all_enrol_X)
            y_enrol   = np.array(all_enrol_y, dtype=int)
            templates = build_templates(cov_enrol, y_enrol)
            tau       = compute_tau(templates, cov_enrol, y_enrol, p)

            all_test_X = [x for sp in day_splits for x in sp["X_test"]]
            all_test_y = [cls2idx[sp["participant"]] for sp in day_splits
                          for _ in sp["X_test"]]
            y_test = np.array(all_test_y, dtype=int)

            # deterministic torch seed per (p, day) so runs are reproducible
            torch_seed = SEED + p * 31 + abs(hash(day)) % 10_000

            logs, per_tgt = run_vae_attack(
                templates, tau,
                all_test_X, y_test,
                N_ITER, rng,
                lam=LAMBDA, device=DEVICE,
                seed=torch_seed,
            )
            if not logs:
                continue

            for cls_idx, accepted_list in per_tgt.items():
                if accepted_list:
                    per_subj_far.append(float(np.mean(accepted_list)))

            for l in logs:
                if l["accepted"]:
                    budget_accepted.append(l["budget"])

            if p == 10:
                rolling_far.extend([int(l["accepted"]) for l in logs])

        mean_far    = float(np.mean(per_subj_far))    if per_subj_far    else 0.0
        std_far     = float(np.std(per_subj_far))     if per_subj_far    else 0.0
        mean_budget = float(np.mean(budget_accepted)) if budget_accepted else float("nan")

        all_results[p] = dict(
            mean_far    = mean_far,
            std_far     = std_far,
            mean_budget = mean_budget,
            rolling_far = rolling_far if p == 10 else [],
        )
        print(f"  p={p:>2}: FAR = {mean_far:.3f} ± {std_far:.3f}   "
              f"budget = {mean_budget:.3f}")

    # ── save CSV ──────────────────────────────────────────────────────────────
    rows = [dict(p=p,
                 FAR_mean=all_results[p]["mean_far"],
                 FAR_std =all_results[p]["std_far"],
                 Budget_mean=all_results[p]["mean_budget"])
            for p in P_ATTACK]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "adv_far_vae.csv"), index=False)

    # ── convergence plot (p=10) ───────────────────────────────────────────────
    rf = all_results.get(10, {}).get("rolling_far", [])
    if rf:
        _plot_convergence(rf, ds_name, out_dir)

    return all_results


# ── plots ─────────────────────────────────────────────────────────────────────

def _plot_convergence(rolling_far: list, ds_name: str, out_dir: str,
                      window: int = 100) -> None:
    arr      = np.array(rolling_far, dtype=float)
    kernel   = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    x        = np.arange(len(smoothed)) + window

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, smoothed, color="darkorange", lw=1.5, label="VAE rolling FAR")
    ax.axhline(arr.mean(), color="gray", ls="--", lw=1,
               label=f"mean FAR = {arr.mean():.3f}")
    ax.set_xlabel("Iteration (pooled across days)", fontsize=12)
    ax.set_ylabel(f"Rolling FAR (window={window})", fontsize=12)
    ax.set_title(f"{ds_name} — VAE attack convergence at $p=10$", fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence_p10.png"), dpi=150)
    plt.close()


def _plot_comparison(summary_vae: dict, summary_ucb: dict,
                     out_dir: str) -> None:
    """Bar chart: UCB MAB vs VAE at p=10 for all 8 datasets."""
    ds_order = ["zhang", "won",
                "tanner_eo", "tanner_ec", "cogbci_eo", "cogbci_ec",
                "cho", "lee"]
    ds_names = [d for d in ds_order if d in summary_vae]

    vae_far = [summary_vae[d].get(10, {}).get("mean_far", 0.0) for d in ds_names]
    vae_std = [summary_vae[d].get(10, {}).get("std_far",  0.0) for d in ds_names]
    ucb_far = [summary_ucb.get(d, {}).get(10, {}).get("mean_far", float("nan"))
               for d in ds_names]
    ucb_std = [summary_ucb.get(d, {}).get(10, {}).get("std_far",  0.0)
               for d in ds_names]

    x, w = np.arange(len(ds_names)), 0.38
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w/2, ucb_far, w, yerr=ucb_std, capsize=4,
           color="steelblue", alpha=0.85, label="UCB MAB")
    ax.bar(x + w/2, vae_far, w, yerr=vae_std, capsize=4,
           color="darkorange", alpha=0.85, label="VAE generator")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in ds_names], fontsize=9)
    ax.set_ylabel("Adversarial FAR (mean ± std)", fontsize=12)
    ax.set_title("UCB MAB vs. VAE perturbation generator — adversarial FAR at $p=10$",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "vae_vs_ucb_p10.png"), dpi=150)
    plt.close()
    print(f"Saved → {out_dir}/vae_vs_ucb_p10.png")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    summary_vae = {}

    for ds_name, cfg in DATASETS.items():
        print(f"\n{'='*64}\n  {ds_name.upper()}  (VAE)\n{'='*64}")
        os.makedirs(cfg["out_dir"], exist_ok=True)

        sessions = load_sessions(cfg["participants"], cfg["days"],
                                 erp_root=cfg["erp_root"])
        print(f"Sessions: {len(sessions)}  "
              f"(subjects: {len({s['participant'] for s in sessions})})")
        if not sessions:
            print("  No sessions found — skipping.")
            continue

        summary_vae[ds_name] = run_vae_dataset(sessions, cfg["out_dir"], ds_name)

    # ── load UCB results from existing CSVs ───────────────────────────────────
    summary_ucb = {}
    for ds_name, cfg in UCB_DATASETS.items():
        csv_path = os.path.join(cfg["out_dir"], "adv_far_std.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        summary_ucb[ds_name] = {}
        for _, row in df.iterrows():
            p_val = int(row["p"])
            if p_val in P_ATTACK:
                summary_ucb[ds_name][p_val] = dict(
                    mean_far = float(row["FAR_mean"]),
                    std_far  = float(row["FAR_std"]),
                )

    # ── comparison plot and CSV ───────────────────────────────────────────────
    os.makedirs("results_vae_comparison", exist_ok=True)
    if summary_vae and summary_ucb:
        _plot_comparison(summary_vae, summary_ucb, "results_vae_comparison")

    # full comparison table across all p values
    rows = []
    for ds in DATASETS:
        for p in P_ATTACK:
            vae_m = summary_vae.get(ds, {}).get(p, {}).get("mean_far", float("nan"))
            vae_s = summary_vae.get(ds, {}).get(p, {}).get("std_far",  float("nan"))
            vae_b = summary_vae.get(ds, {}).get(p, {}).get("mean_budget", float("nan"))
            ucb_m = summary_ucb.get(ds, {}).get(p, {}).get("mean_far", float("nan"))
            ucb_s = summary_ucb.get(ds, {}).get(p, {}).get("std_far",  float("nan"))
            rows.append(dict(dataset=ds, p=p,
                             UCB_FAR_mean=ucb_m, UCB_FAR_std=ucb_s,
                             VAE_FAR_mean=vae_m, VAE_FAR_std=vae_s,
                             VAE_Budget_mean=vae_b,
                             Delta_FAR=vae_m - ucb_m))
    pd.DataFrame(rows).to_csv(
        "results_vae_comparison/vae_vs_ucb_full.csv", index=False)

    # ── terminal summary ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 72)
    print(f"{'Dataset':<20}  {'UCB p=10':>12}  {'VAE p=10':>12}  {'Δ':>7}")
    print("=" * 72)
    for ds in DATASETS:
        ucb_m = summary_ucb.get(ds, {}).get(10, {}).get("mean_far", float("nan"))
        ucb_s = summary_ucb.get(ds, {}).get(10, {}).get("std_far",  float("nan"))
        vae_m = summary_vae.get(ds, {}).get(10, {}).get("mean_far", float("nan"))
        vae_s = summary_vae.get(ds, {}).get(10, {}).get("std_far",  float("nan"))
        delta = vae_m - ucb_m
        print(f"{ds:<20}  {ucb_m:.3f}±{ucb_s:.3f}    {vae_m:.3f}±{vae_s:.3f}  "
              f"{delta:+.3f}")

    print("\nDone.")
