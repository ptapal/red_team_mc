"""
Surrogate-gradient GAN attack — all 8 datasets.

True minimax: G and D fight via backprop.  S is a frozen surrogate trained
on enrollment data that makes the Riemannian authenticator differentiable.

Conditions shared with UCB / VAE / pseudo-GAN evaluations:
  - Same subject subsamples, days, enrollment/test splits (SEED=42, rho=0.5)
  - Same dual-reward weight λ=0.3 (authenticator/budget tradeoff in G's loss)
  - Same T=1000 iterations per (day, p) combination
  - FAR measured on the real Riemannian authenticator (not S)

Output per dataset
------------------
  results_sgan_{ds}/adv_far_sgan.csv    — FAR mean±std + mean budget
  results_sgan_{ds}/convergence_p10.png — rolling FAR vs. iteration
  results_sgan_{ds}/defense_p10.csv     — D detection metrics at p=10

Comparison output
-----------------
  results_sgan_comparison/sgan_vs_others_p10.png  — 4-way bar: UCB/VAE/GAN/SGAN
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
import torch

warnings.filterwarnings("ignore")

from helpers  import load_sessions
from config   import SEED
from bandit_all import (
    WON_15, TANNER_20, CHO_20, LEE_20,
    TANNER_DAYS, COGBCI_DAYS, COGBCI_SUBS, CHO_DAYS, LEE_DAYS,
    DATASETS as UCB_DATASETS,
    make_splits, pool_enrol, build_templates, compute_tau, cov_array,
    P_ATTACK,
)
from attacks.sgan import (
    run_surrogate_gan_attack, eval_discriminator, generate_perturbed,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAMBDA = 0.3
N_ITER = 1000
N_DEFENSE_EVAL = 200   # raw trials scored by D for defense metrics

DATASETS = {
    "zhang": dict(
        erp_root     = "/Users/a1234/red_team_mc/data/preprocessed/zhang/won_channels",
        participants = [f"S{i}" for i in range(1, 16)],
        days         = ["Day_1", "Day_7", "Day_80", "Day_200"],
        out_dir      = "results_sgan_zhang",
    ),
    "won": dict(
        erp_root     = "/Users/a1234/red_team_mc/data/preprocessed/won",
        participants = WON_15,
        days         = ["Day_1"],
        out_dir      = "results_sgan_won",
    ),
    "tanner_eo": dict(
        erp_root     = "/Users/a1234/red_team_mc/data/preprocessed/wang/eo",
        participants = TANNER_20,
        days         = TANNER_DAYS,
        out_dir      = "results_sgan_tanner_eo",
    ),
    "tanner_ec": dict(
        erp_root     = "/Users/a1234/red_team_mc/data/preprocessed/wang/ec",
        participants = TANNER_20,
        days         = TANNER_DAYS,
        out_dir      = "results_sgan_tanner_ec",
    ),
    "cogbci_eo": dict(
        erp_root     = "/Users/a1234/red_team_mc/data/preprocessed/cogbci/eo",
        participants = COGBCI_SUBS,
        days         = COGBCI_DAYS,
        out_dir      = "results_sgan_cogbci_eo",
    ),
    "cogbci_ec": dict(
        erp_root     = "/Users/a1234/red_team_mc/data/preprocessed/cogbci/ec",
        participants = COGBCI_SUBS,
        days         = COGBCI_DAYS,
        out_dir      = "results_sgan_cogbci_ec",
    ),
    "cho": dict(
        erp_root     = "/Users/a1234/red_team_mc/data/preprocessed/cho",
        participants = CHO_20,
        days         = CHO_DAYS,
        out_dir      = "results_sgan_cho",
    ),
    "lee": dict(
        erp_root     = "/Users/a1234/red_team_mc/data/preprocessed/lee",
        participants = LEE_20,
        days         = LEE_DAYS,
        out_dir      = "results_sgan_lee",
    ),
}



def run_sgan_dataset(sessions: list, out_dir: str, ds_name: str) -> dict:
    rng = np.random.default_rng(SEED)
    all_participants = sorted({s["participant"] for s in sessions})
    cls2idx  = {p: i for i, p in enumerate(all_participants)}
    splits   = make_splits(sessions, rng)
    by_day   = pool_enrol(splits)
    all_results = {}

    # store last day's trained G and D at p=10 for defense evaluation
    last_G, last_D, last_test_X = None, None, None

    for p in P_ATTACK:
        per_subj_far    = []
        budget_accepted = []
        rolling_far     = []

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

            torch_seed = SEED + p * 31 + abs(hash(day)) % 10_000

            # run_surrogate_gan_attack returns (logs, per_tgt, D, S, G)
            logs, per_tgt, D, S, G = run_surrogate_gan_attack(
                templates, tau,
                all_test_X, y_test,
                all_enrol_X, y_enrol,
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
                last_G, last_D, last_test_X = G, D, all_test_X

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

    rows = [dict(p=p,
                 FAR_mean=all_results[p]["mean_far"],
                 FAR_std =all_results[p]["std_far"],
                 Budget_mean=all_results[p]["mean_budget"])
            for p in P_ATTACK]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "adv_far_sgan.csv"), index=False)

    rf = all_results.get(10, {}).get("rolling_far", [])
    if rf:
        _plot_convergence(rf, ds_name, out_dir)

    # Use the trained G from the last day's p=10 run to generate perturbed samples,
    # then evaluate D's ability to separate genuine vs. adversarial raw EEG.
    if last_G is not None and last_D is not None and last_test_X:
        n = min(N_DEFENSE_EVAL, len(last_test_X))
        genuine_eval   = last_test_X[:n]
        perturbed_eval = generate_perturbed(last_G, genuine_eval, device=DEVICE)

        def_metrics = eval_discriminator(
            last_D, genuine_eval, perturbed_eval, device=DEVICE
        )
        pd.DataFrame([def_metrics]).to_csv(
            os.path.join(out_dir, "defense_p10.csv"), index=False
        )
        print(f"  Defense (D): TPR={def_metrics['tpr']:.3f}  "
              f"TNR={def_metrics['tnr']:.3f}  "
              f"D(genuine)={def_metrics['mean_d_genuine']:.3f}  "
              f"D(perturbed)={def_metrics['mean_d_perturbed']:.3f}")

    return all_results



def _plot_convergence(rolling_far: list, ds_name: str, out_dir: str,
                      window: int = 100) -> None:
    arr      = np.array(rolling_far, dtype=float)
    kernel   = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    x        = np.arange(len(smoothed)) + window

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, smoothed, color="purple", lw=1.5, label="SGAN rolling FAR")
    ax.axhline(arr.mean(), color="gray", ls="--", lw=1,
               label=f"mean FAR = {arr.mean():.3f}")
    ax.set_xlabel("Iteration (pooled across days)", fontsize=12)
    ax.set_ylabel(f"Rolling FAR (window={window})", fontsize=12)
    ax.set_title(f"{ds_name} — surrogate GAN convergence at $p=10$", fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence_p10.png"), dpi=150)
    plt.close()


def _plot_comparison(summary_sgan: dict, summary_vae: dict,
                     summary_gan: dict,  summary_ucb: dict,
                     out_dir: str) -> None:
    ds_order = ["zhang", "won",
                "tanner_eo", "tanner_ec", "cogbci_eo", "cogbci_ec",
                "cho", "lee"]
    ds_names = [d for d in ds_order if d in summary_sgan]

    def _m(s, ds): return s.get(ds, {}).get(10, {}).get("mean_far", float("nan"))
    def _e(s, ds): return s.get(ds, {}).get(10, {}).get("std_far",  0.0)

    x, w = np.arange(len(ds_names)), 0.2
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (label, summary, color) in enumerate([
        ("UCB MAB",    summary_ucb,  "steelblue"),
        ("VAE",        summary_vae,  "darkorange"),
        ("pseudo-GAN", summary_gan,  "seagreen"),
        ("SGAN",       summary_sgan, "purple"),
    ]):
        vals = [_m(summary, d) for d in ds_names]
        errs = [_e(summary, d) for d in ds_names]
        ax.bar(x + (i - 1.5) * w, vals, w, yerr=errs, capsize=3,
               color=color, alpha=0.85, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in ds_names], fontsize=9)
    ax.set_ylabel("Adversarial FAR (mean ± std)", fontsize=12)
    ax.set_title(
        "UCB / VAE / pseudo-GAN / surrogate-GAN — adversarial FAR at $p=10$",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sgan_vs_others_p10.png"), dpi=150)
    plt.close()
    print(f"Saved → {out_dir}/sgan_vs_others_p10.png")



if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    summary_sgan = {}

    for ds_name, cfg in DATASETS.items():
        print(f"\n{'='*64}\n  {ds_name.upper()}  (Surrogate GAN)\n{'='*64}")
        os.makedirs(cfg["out_dir"], exist_ok=True)

        sessions = load_sessions(cfg["participants"], cfg["days"],
                                 erp_root=cfg["erp_root"])
        print(f"Sessions: {len(sessions)}  "
              f"(subjects: {len({s['participant'] for s in sessions})})")
        if not sessions:
            print("  No sessions found — skipping.")
            continue

        summary_sgan[ds_name] = run_sgan_dataset(
            sessions, cfg["out_dir"], ds_name
        )

    def _load(out_dir, fname):
        path = os.path.join(out_dir, fname)
        if not os.path.exists(path):
            return {}
        df = pd.read_csv(path)
        out = {}
        for _, row in df.iterrows():
            p = int(row["p"])
            if p in P_ATTACK:
                out[p] = dict(mean_far=float(row["FAR_mean"]),
                              std_far =float(row["FAR_std"]))
        return out

    summary_ucb = {ds: _load(cfg["out_dir"], "adv_far_std.csv")
                   for ds, cfg in UCB_DATASETS.items()}
    summary_vae = {ds: _load(f"results_vae_{ds}", "adv_far_vae.csv")
                   for ds in DATASETS}
    summary_gan = {ds: _load(f"results_gan_{ds}", "adv_far_gan.csv")
                   for ds in DATASETS}

    os.makedirs("results_sgan_comparison", exist_ok=True)
    _plot_comparison(summary_sgan, summary_vae, summary_gan, summary_ucb,
                     "results_sgan_comparison")

    print("\n\n" + "=" * 96)
    print(f"{'Dataset':<20}  {'UCB p=10':>12}  {'VAE p=10':>12}  "
          f"{'GAN p=10':>12}  {'SGAN p=10':>12}  ΔSGAN-UCB")
    print("=" * 96)
    for ds in DATASETS:
        def _m(s): return s.get(ds, {}).get(10, {}).get("mean_far", float("nan"))
        def _e(s): return s.get(ds, {}).get(10, {}).get("std_far",  float("nan"))
        um, vm, gm, sm = _m(summary_ucb), _m(summary_vae), _m(summary_gan), _m(summary_sgan)
        ue, ve, ge, se = _e(summary_ucb), _e(summary_vae), _e(summary_gan), _e(summary_sgan)
        delta = sm - um
        print(f"{ds:<20}  {um:.3f}±{ue:.3f}    {vm:.3f}±{ve:.3f}    "
              f"{gm:.3f}±{ge:.3f}    {sm:.3f}±{se:.3f}  {delta:+.3f}")

    print("\nDone.")
