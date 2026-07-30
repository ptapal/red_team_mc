"""
Normalised multi-dataset comparison.
Zhang:       15 subjects, 4 days (ERP / face-RSVP).
Won:         15 subjects subsampled from 55 (fixed SEED), 1 day (ERP / P300-speller).
Tanner EO:  20 subjects subsampled from 60 (seed 42), 3 sessions (resting-state, eyes open).
Tanner EC:  20 subjects subsampled from 60 (seed 42), 3 sessions (resting-state, eyes closed).
COG-BCI EO: 20 subjects (sub-10..sub-29), 3 sessions (resting-state, eyes open).
COG-BCI EC: 20 subjects (sub-10..sub-29), 3 sessions (resting-state, eyes closed).
Cho:         20 subjects subsampled from 52 (seed 42), 1 session (motor imagery).
Lee:         20 subjects subsampled from 54 (seed 42), 2 sessions (motor imagery).
Metrics reported as mean ± std across subjects.
"""

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

ENROL_FRAC = 0.5
LAMBDA     = 0.3
P_SWEEP    = list(range(2, 51, 2))
P_ATTACK   = [5, 10, 15, 20, 25]
N_ITER     = 1000


# ── subsample 15 Won subjects ──────────────────────────────────────────────────
_rng_s    = np.random.default_rng(SEED)
WON_15    = sorted(
    _rng_s.choice([f"S{i:02d}" for i in range(1, 56)], size=15, replace=False).tolist()
)

TANNER_DAYS  = ["session1", "session2", "session3"]
COGBCI_DAYS  = ["S1", "S2", "S3"]
COGBCI_SUBS  = [f"sub-{i:02d}" for i in range(10, 30)]

# Subsample 20 Tanner subjects from 60 to match COG-BCI N (seed 42)
_rng_t20 = np.random.default_rng(42)
TANNER_20 = sorted(
    _rng_t20.choice([f"sub-{i:02d}" for i in range(1, 61)], size=20, replace=False).tolist()
)

# Subsample 20 Cho subjects from 52 (seed 42)
_rng_cho = np.random.default_rng(42)
CHO_20 = sorted(
    _rng_cho.choice([f"s{i:02d}" for i in range(1, 53)], size=20, replace=False).tolist()
)

# Subsample 20 Lee subjects from 54 (seed 42)
_rng_lee = np.random.default_rng(42)
LEE_20 = sorted(
    _rng_lee.choice([f"sub-{i:02d}" for i in range(1, 55)], size=20, replace=False).tolist()
)

CHO_DAYS = ["session1"]
LEE_DAYS = ["session1", "session2"]

DATASETS = {
    "zhang": dict(
        erp_root    = "/Users/a1234/red_team_mc/data/preprocessed/zhang/won_channels",
        participants= [f"S{i}" for i in range(1, 16)],
        days        = ["Day_1", "Day_7", "Day_80", "Day_200"],
        out_dir     = "results_zhang",
    ),
    "won": dict(
        erp_root    = "/Users/a1234/red_team_mc/data/preprocessed/won",
        participants= WON_15,
        days        = ["Day_1"],
        out_dir     = "results_won",
    ),
    "tanner_eo": dict(
        erp_root    = "/Users/a1234/red_team_mc/data/preprocessed/wang/eo",
        participants= TANNER_20,
        days        = TANNER_DAYS,
        out_dir     = "results_tanner_eo",
    ),
    "tanner_ec": dict(
        erp_root    = "/Users/a1234/red_team_mc/data/preprocessed/wang/ec",
        participants= TANNER_20,
        days        = TANNER_DAYS,
        out_dir     = "results_tanner_ec",
    ),
    "cogbci_eo": dict(
        erp_root    = "/Users/a1234/red_team_mc/data/preprocessed/cogbci/eo",
        participants= COGBCI_SUBS,
        days        = COGBCI_DAYS,
        out_dir     = "results_cogbci_eo",
    ),
    "cogbci_ec": dict(
        erp_root    = "/Users/a1234/red_team_mc/data/preprocessed/cogbci/ec",
        participants= COGBCI_SUBS,
        days        = COGBCI_DAYS,
        out_dir     = "results_cogbci_ec",
    ),
    "cho": dict(
        erp_root    = "/Users/a1234/red_team_mc/data/preprocessed/cho",
        participants= CHO_20,
        days        = CHO_DAYS,
        out_dir     = "results_cho",
    ),
    "lee": dict(
        erp_root    = "/Users/a1234/red_team_mc/data/preprocessed/lee",
        participants= LEE_20,
        days        = LEE_DAYS,
        out_dir     = "results_lee",
    ),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def build_templates(cov_enrol, y_enrol):
    templates = {}
    for cls in np.unique(y_enrol):
        tmpl = cov_enrol[y_enrol == cls].mean(axis=0)
        templates[int(cls)] = tmpl / (np.linalg.norm(tmpl) + 1e-9)
    return templates


def compute_tau(templates, cov_enrol, y_enrol, p):
    genuine = [cosine_sim(v, templates[int(c)])
               for c in np.unique(y_enrol) for v in cov_enrol[y_enrol == c]]
    return float(np.percentile(genuine, p))


def cov_array(trial_list):
    return np.array([trial_to_cov_vec(x) for x in trial_list])


# ── split sessions: per-day, per-subject ──────────────────────────────────────

def make_splits(sessions, rng, enrol_frac=ENROL_FRAC):
    """
    Returns list of split dicts, one per (subject, day).
    Each dict: participant, day, X_enrol (list), y_enrol (int),
               X_test (list), cls (int)
    """
    splits = []
    for sess in sessions:
        X = sess["X"]
        n = len(X)
        perm = rng.permutation(n)
        cut  = max(2, int(n * enrol_frac))
        splits.append(dict(
            participant = sess["participant"],
            day         = sess["day"],
            X_enrol     = [X[i] for i in perm[:cut]],
            X_test      = [X[i] for i in perm[cut:]],
        ))
    return splits


def pool_enrol(splits):
    """Pool all subjects' enrolment for a given day → cov arrays + labels."""
    by_day = {}
    for sp in splits:
        by_day.setdefault(sp["day"], []).append(sp)
    return by_day


# ── threshold sweep with per-subject std ──────────────────────────────────────

def threshold_sweep(sessions, out_dir, ds_name):
    rng  = np.random.default_rng(SEED)
    all_participants = sorted({s["participant"] for s in sessions})
    cls2idx = {p: i for i, p in enumerate(all_participants)}
    splits  = make_splits(sessions, rng)
    by_day  = pool_enrol(splits)

    # per-subject-per-day FRR and FAR at each p
    # observation unit: (subject, day)
    p2frr, p2far = {p: [] for p in P_SWEEP}, {p: [] for p in P_SWEEP}

    for day, day_splits in by_day.items():
        # Build global (day-level) templates from all enrolment data
        all_enrol_X  = [x for sp in day_splits for x in sp["X_enrol"]]
        all_enrol_y  = [cls2idx[sp["participant"]] for sp in day_splits
                        for _ in sp["X_enrol"]]
        cov_enrol_all  = cov_array(all_enrol_X)
        y_enrol_all    = np.array(all_enrol_y, dtype=int)
        templates_all  = build_templates(cov_enrol_all, y_enrol_all)

        for p in P_SWEEP:
            tau = compute_tau(templates_all, cov_enrol_all, y_enrol_all, p)
            classes = list(templates_all.keys())

            for sp in day_splits:
                cls  = cls2idx[sp["participant"]]
                if cls not in templates_all or not sp["X_test"]:
                    continue
                test_vecs = cov_array(sp["X_test"])
                gen_scores = [cosine_sim(v, templates_all[cls]) for v in test_vecs]
                frr = float(np.mean(np.array(gen_scores) < tau))
                p2frr[p].append(frr)

                others = [c for c in classes if c != cls]
                if others:
                    imp_cls = int(rng.choice(others))
                    imp_scores = [cosine_sim(v, templates_all[imp_cls]) for v in test_vecs]
                    far = float(np.mean(np.array(imp_scores) >= tau))
                    p2far[p].append(far)

    # aggregate
    results = {}
    for p in P_SWEEP:
        results[p] = dict(
            mean_frr = np.mean(p2frr[p]),  std_frr = np.std(p2frr[p]),
            mean_far  = np.mean(p2far[p]),  std_far  = np.std(p2far[p]),
        )

    # EER
    diffs   = [abs(results[p]["mean_frr"] - results[p]["mean_far"]) for p in P_SWEEP]
    eer_p   = P_SWEEP[np.argmin(diffs)]
    eer_val = (results[eer_p]["mean_frr"] + results[eer_p]["mean_far"]) / 2

    # print
    print(f"\n{ds_name} threshold sweep (mean±std across subjects)")
    print(f"{'p':>4}  {'FRR mean±std':>18}  {'FAR mean±std':>18}")
    print("-" * 46)
    for p in P_SWEEP:
        r = results[p]
        mark = "  ← EER" if p == eer_p else ("  ← ref" if p == 10 else "")
        print(f"{p:>4}  {r['mean_frr']:.3f}±{r['std_frr']:.3f}          "
              f"{r['mean_far']:.3f}±{r['std_far']:.3f}{mark}")
    print(f"EER ≈ {eer_val:.3f} at p = {eer_p}")

    # save csv
    rows = [dict(p=p, FRR_mean=results[p]["mean_frr"], FRR_std=results[p]["std_frr"],
                 FAR_mean=results[p]["mean_far"],  FAR_std=results[p]["std_far"])
            for p in P_SWEEP]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "threshold_sweep_std.csv"), index=False)

    # plot
    ps       = P_SWEEP
    frr_m    = [results[p]["mean_frr"] for p in ps]
    far_m    = [results[p]["mean_far"]  for p in ps]
    frr_s    = [results[p]["std_frr"]  for p in ps]
    far_s    = [results[p]["std_far"]   for p in ps]
    fig, ax  = plt.subplots(figsize=(8, 5))
    ax.plot(ps, frr_m, "o-", color="steelblue", label="FRR")
    ax.fill_between(ps, np.array(frr_m)-np.array(frr_s),
                        np.array(frr_m)+np.array(frr_s), color="steelblue", alpha=0.15)
    ax.plot(ps, far_m, "s-", color="tomato", label="FAR")
    ax.fill_between(ps, np.array(far_m)-np.array(far_s),
                        np.array(far_m)+np.array(far_s), color="tomato", alpha=0.15)
    ax.axvline(eer_p, color="green", ls=":", lw=1.4,
               label=f"EER $\\approx$ {eer_val:.3f} ($p={eer_p}$)")
    ax.axvline(10, color="gray", ls="--", lw=1.2, label="$p=10$")
    ax.set_xlabel("Threshold percentile $p$", fontsize=13)
    ax.set_ylabel("Rate", fontsize=13)
    ax.set_title(f"{ds_name} — FRR/FAR vs. $p$ (±1 std across subjects)", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "threshold_sweep_std.png"), dpi=150)
    plt.close()

    return results, eer_p, eer_val


# ── UCB adversarial attack with per-subject std ───────────────────────────────

def _attack_loop(templates, tau, X_test, y_test, n_iter, rng, lam=LAMBDA):
    """Returns logs list and per-target-class accepted-fraction dict."""
    X_arr   = np.array(X_test)
    classes = np.array(list(templates.keys()))
    logs    = []
    per_tgt = {int(c): [] for c in classes}

    ucb  = UCBSelector(n_arms=N_ARMS, c=DEFAULT_C_UCB)
    arms = AttackArms(rng=rng, s_freq=250)

    for _ in range(n_iter):
        arm = ucb.select()
        vi  = int(rng.integers(len(X_arr)))
        X0  = X_arr[vi]
        ts  = int(y_test[vi])

        imp_trial = None
        if arm == 4:
            others = [c for c in classes if c != ts]
            if len(others):
                ic = int(rng.choice(others))
                im = np.where(y_test == ic)[0]
                if len(im):
                    imp_trial = X_arr[int(rng.choice(im))]
        try:
            Xp, _ = arms.apply(arm, X0, imp_trial)
        except Exception:
            ucb.update(arm, 0.0); continue

        others = [c for c in classes if c != ts]
        tgt    = int(rng.choice(others)) if len(others) else ts
        if tgt not in templates:
            ucb.update(arm, 0.0); continue

        q        = trial_to_cov_vec(Xp)
        score    = cosine_sim(q, templates[tgt])
        accepted = score >= tau
        delta    = float(np.linalg.norm(Xp - X0) / (np.linalg.norm(X0) + 1e-9))
        reward   = (1 - lam) * float(accepted) + lam * max(0.0, 1.0 - delta)
        ucb.update(arm, reward)
        logs.append(dict(arm=arm, arm_name=ARM_NAMES[arm],
                         tgt=tgt, accepted=bool(accepted), reward=float(reward)))
        per_tgt[tgt].append(bool(accepted))

    return logs, per_tgt, ucb


def run_ucb_attack(sessions, out_dir, ds_name):
    """Returns: p -> {mean_far, std_far, arm: {name: (mean, std)}}"""
    rng = np.random.default_rng(SEED)
    all_participants = sorted({s["participant"] for s in sessions})
    cls2idx  = {p: i for i, p in enumerate(all_participants)}
    splits   = make_splits(sessions, rng)
    by_day   = pool_enrol(splits)
    all_results = {}

    for p in P_ATTACK:
        day_far_vals  = []
        arm_logs_all  = {name: [] for name in ARM_NAMES}
        per_subj_far  = []  # one value per (subject, day)

        for day, day_splits in sorted(by_day.items()):
            # global templates for this day
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

            logs, per_tgt, ucb = _attack_loop(
                templates, tau, all_test_X, y_test, N_ITER, rng)
            if not logs:
                continue

            day_far = float(np.mean([l["accepted"] for l in logs]))
            day_far_vals.append(day_far)

            # per-subject FAR: fraction of attempts targeting each subject's identity
            for cls_idx, accepted_list in per_tgt.items():
                if accepted_list:
                    per_subj_far.append(float(np.mean(accepted_list)))

            # per-arm FAR
            for name in ARM_NAMES:
                arm_l = [l for l in logs if l["arm_name"] == name]
                if arm_l:
                    arm_logs_all[name].append(float(np.mean([l["accepted"] for l in arm_l])))

        all_results[p] = dict(
            mean_far    = float(np.mean(per_subj_far)) if per_subj_far else 0.0,
            std_far     = float(np.std(per_subj_far))  if per_subj_far else 0.0,
            mean_far_day= float(np.mean(day_far_vals)) if day_far_vals else 0.0,
            arm={name: (float(np.mean(v)) if v else 0.0,
                        float(np.std(v))  if v else 0.0)
                 for name, v in arm_logs_all.items()},
        )

        r = all_results[p]
        print(f"\n{ds_name} p={p}: adversarial FAR = {r['mean_far']:.3f}±{r['std_far']:.3f}")
        print(f"  {'Arm':<26}  {'FAR mean±std':>18}")
        for name in ARM_NAMES:
            m, s = r["arm"][name]
            print(f"  {name:<26}  {m:.3f}±{s:.3f}")
        best = max(r["arm"], key=lambda n: r["arm"][n][0])
        print(f"  Best arm: {best}")

    # arm pull plot
    _plot_arms(all_results, ds_name, out_dir)

    # save csv
    rows = []
    for p in P_ATTACK:
        r = all_results[p]
        row = dict(p=p, FAR_mean=r["mean_far"], FAR_std=r["std_far"])
        for name in ARM_NAMES:
            row[f"arm_{name}_mean"], row[f"arm_{name}_std"] = r["arm"][name]
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "adv_far_std.csv"), index=False)

    return all_results


def _plot_arms(adv_results, ds_name, out_dir):
    p_ref = 10
    if p_ref not in adv_results:
        return
    r    = adv_results[p_ref]
    means = [r["arm"][n][0] for n in ARM_NAMES]
    stds  = [r["arm"][n][1] for n in ARM_NAMES]
    idx   = np.argsort(means)[::-1]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(ARM_NAMES)), [means[i] for i in idx], color="steelblue",
           yerr=[stds[i] for i in idx], capsize=4, alpha=0.85)
    ax.set_xticks(range(len(ARM_NAMES)))
    ax.set_xticklabels([ARM_NAMES[i] for i in idx], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Adversarial FAR", fontsize=12)
    ax.set_title(f"{ds_name} — per-arm adversarial FAR at $p=10$ (±1 std)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "arm_far_std.png"), dpi=150)
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Won 15 subjects (seed={SEED}): {WON_15}")
    print(f"Tanner 20 subjects (seed=42): {TANNER_20}")
    print(f"Cho 20 subjects (seed=42): {CHO_20}")
    print(f"Lee 20 subjects (seed=42): {LEE_20}\n")

    summary = {}

    for ds_name, cfg in DATASETS.items():
        print(f"\n{'='*64}\n  {ds_name.upper()}\n{'='*64}")
        os.makedirs(cfg["out_dir"], exist_ok=True)

        sessions = load_sessions(cfg["participants"], cfg["days"],
                                 erp_root=cfg["erp_root"])
        print(f"Sessions loaded: {len(sessions)}  "
              f"(subjects: {len({s['participant'] for s in sessions})})")

        sw, eer_p, eer = threshold_sweep(sessions, cfg["out_dir"], ds_name)
        adv            = run_ucb_attack(sessions,  cfg["out_dir"], ds_name)

        summary[ds_name] = dict(sweep=sw, eer_p=eer_p, eer=eer, adv=adv)

    print("\n\n" + "="*64)
    print("SUMMARY")
    print("="*64)

    for ds in ["zhang", "won", "tanner_eo", "tanner_ec", "cogbci_eo", "cogbci_ec", "cho", "lee"]:
        if ds not in summary:
            continue
        s = summary[ds]
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
