import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from helpers  import load_sessions
from utils    import trial_to_cov_vec, cosine_sim
from config   import SEED, N_ARMS, ARM_NAMES, DEFAULT_C_UCB
from arms     import AttackArms
from ucb      import UCBSelector
from datasets import get_datasets
from bandit_all import cov_array, P_SWEEP, P_ATTACK, N_ITER, LAMBDA, build_templates
from run_gnb     import build_gnb_models, gnb_score

CROSS_SESSION_DATASETS = ["zhang", "tanner_eo", "tanner_ec", "cogbci_eo", "cogbci_ec", "lee"]
OUT_ROOT = "results/cross_session"

CLASSIFIERS = {
    "nearest_centroid": dict(build_fn=build_templates, score_fn=cosine_sim),
    "gnb":               dict(build_fn=build_gnb_models, score_fn=gnb_score),
}


def group_by_subject_day(sessions: list) -> dict:
    """participant -> {day: X (trials array)}"""
    out = {}
    for s in sessions:
        out.setdefault(s["participant"], {})[s["day"]] = s["X"]
    return out


def build_pair_splits(subj_days: dict, enroll_day: str, test_day: str, cls2idx: dict):
    """Subjects present in BOTH enroll_day and test_day. Enrollment uses the
    subject's full enroll_day session; test uses the full test_day session."""
    X_enrol, y_enrol, X_test, y_test = [], [], [], []
    for participant, day_data in subj_days.items():
        if enroll_day not in day_data or test_day not in day_data:
            continue
        cls = cls2idx[participant]
        for x in day_data[enroll_day]:
            X_enrol.append(x); y_enrol.append(cls)
        for x in day_data[test_day]:
            X_test.append(x); y_test.append(cls)
    return X_enrol, np.array(y_enrol, dtype=int), X_test, np.array(y_test, dtype=int)


def compute_tau_generic(models, score_fn, cov_enrol, y_enrol, p):
    genuine = [score_fn(models[int(c)], v)
               for c in np.unique(y_enrol) for v in cov_enrol[y_enrol == c]]
    return float(np.percentile(genuine, p))


def threshold_sweep_pair(models, score_fn, cov_enrol, y_enrol, cov_test, y_test):
    classes = list(models.keys())
    p2frr, p2far = {}, {}
    for p in P_SWEEP:
        tau = compute_tau_generic(models, score_fn, cov_enrol, y_enrol, p)
        frrs, fars = [], []
        for cls in classes:
            mask = (y_test == cls)
            if not mask.any():
                continue
            test_vecs = cov_test[mask]
            gen_scores = [score_fn(models[cls], v) for v in test_vecs]
            frrs.append(float(np.mean(np.array(gen_scores) < tau)))
            others = [c for c in classes if c != cls]
            if others:
                imp_cls = int(np.random.default_rng(SEED).choice(others))
                imp_scores = [score_fn(models[imp_cls], v) for v in test_vecs]
                fars.append(float(np.mean(np.array(imp_scores) >= tau)))
        p2frr[p] = float(np.mean(frrs)) if frrs else float("nan")
        p2far[p] = float(np.mean(fars)) if fars else float("nan")

    diffs   = [abs(p2frr[p] - p2far[p]) for p in P_SWEEP]
    eer_p   = P_SWEEP[int(np.argmin(diffs))]
    eer_val = (p2frr[eer_p] + p2far[eer_p]) / 2
    return p2frr, p2far, eer_p, eer_val


def attack_loop_generic(models, score_fn, tau, X_test, y_test, n_iter, rng, lam=LAMBDA):
    X_arr   = np.array(X_test)
    classes = np.array(list(models.keys()))
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
        if tgt not in models:
            ucb.update(arm, 0.0); continue

        q        = trial_to_cov_vec(Xp)
        score    = score_fn(models[tgt], q)
        accepted = score >= tau
        delta    = float(np.linalg.norm(Xp - X0) / (np.linalg.norm(X0) + 1e-9))
        reward   = (1 - lam) * float(accepted) + lam * max(0.0, 1.0 - delta)
        ucb.update(arm, reward)
        logs.append(dict(arm_name=ARM_NAMES[arm], accepted=bool(accepted)))
        per_tgt[tgt].append(bool(accepted))

    return logs, per_tgt


def eval_pair(clf_name, ds_name, enroll_day, test_day, X_enrol, y_enrol, X_test, y_test, out_dir):
    build_fn = CLASSIFIERS[clf_name]["build_fn"]
    score_fn = CLASSIFIERS[clf_name]["score_fn"]

    cov_enrol = cov_array(X_enrol)
    models    = build_fn(cov_enrol, y_enrol)
    cov_test  = cov_array(X_test)

    p2frr, p2far, eer_p, eer_val = threshold_sweep_pair(
        models, score_fn, cov_enrol, y_enrol, cov_test, y_test)
    print(f"  [{clf_name}/{ds_name}] {enroll_day} -> {test_day}: "
          f"EER={eer_val:.3f} at p={eer_p}  "
          f"(n_enrol={len(X_enrol)}, n_test={len(X_test)}, subjects={len(set(y_test))})")

    pd.DataFrame([dict(p=p, FRR=p2frr[p], FAR=p2far[p]) for p in P_SWEEP]).to_csv(
        os.path.join(out_dir, "threshold_sweep.csv"), index=False)

    rng = np.random.default_rng(SEED)
    adv_rows = []
    for p in P_ATTACK:
        tau = compute_tau_generic(models, score_fn, cov_enrol, y_enrol, p)
        atk_seed = SEED + p * 31 + abs(hash(f"{enroll_day}->{test_day}")) % 10_000
        rng_atk  = np.random.default_rng(atk_seed)
        logs, per_tgt = attack_loop_generic(
            models, score_fn, tau, X_test, y_test, N_ITER, rng_atk)

        per_subj_far = [float(np.mean(v)) for v in per_tgt.values() if v]
        mean_far = float(np.mean(per_subj_far)) if per_subj_far else 0.0
        std_far  = float(np.std(per_subj_far))  if per_subj_far else 0.0

        arm_means = {}
        for name in ARM_NAMES:
            arm_l = [l["accepted"] for l in logs if l["arm_name"] == name]
            arm_means[name] = float(np.mean(arm_l)) if arm_l else 0.0
        best_arm = max(arm_means, key=arm_means.get)

        print(f"    p={p:>2}: adv FAR={mean_far:.3f}+-{std_far:.3f}  best={best_arm}")
        row = dict(p=p, FAR_mean=mean_far, FAR_std=std_far, best_arm=best_arm)
        row.update({f"arm_{k}": v for k, v in arm_means.items()})
        adv_rows.append(row)

    pd.DataFrame(adv_rows).to_csv(os.path.join(out_dir, "adv_far.csv"), index=False)

    return dict(eer=eer_val, eer_p=eer_p,
                base_far_p10=p2far.get(10, float("nan")),
                base_frr_p10=p2frr.get(10, float("nan")),
                adv_far_p10=next((r["FAR_mean"] for r in adv_rows if r["p"] == 10), float("nan")),
                best_arm_p10=next((r["best_arm"] for r in adv_rows if r["p"] == 10), None))


if __name__ == "__main__":
    DATASETS = get_datasets("results")
    summary_rows = []

    for ds_name in CROSS_SESSION_DATASETS:
        cfg = DATASETS[ds_name]
        print(f"\n{'='*64}\n  {ds_name.upper()}  (cross-session)\n{'='*64}")
        sessions = load_sessions(cfg["participants"], cfg["days"], erp_root=cfg["erp_root"])
        if not sessions:
            print("  No sessions -- skipping.")
            continue

        subj_days = group_by_subject_day(sessions)
        all_participants = sorted(subj_days.keys())
        cls2idx = {p: i for i, p in enumerate(all_participants)}

        days_order = cfg["days"]  
        enroll_day = days_order[0]
        test_days  = days_order[1:]

        for clf_name in CLASSIFIERS:
            for test_day in test_days:
                X_enrol, y_enrol, X_test, y_test = build_pair_splits(
                    subj_days, enroll_day, test_day, cls2idx)
                if len(X_enrol) == 0 or len(X_test) == 0:
                    print(f"  [{clf_name}] {enroll_day} -> {test_day}: no overlapping subjects, skipping.")
                    continue

                out_dir = os.path.join(OUT_ROOT, clf_name, ds_name, f"{enroll_day}_to_{test_day}")
                os.makedirs(out_dir, exist_ok=True)

                res = eval_pair(clf_name, ds_name, enroll_day, test_day,
                                X_enrol, y_enrol, X_test, y_test, out_dir)
                summary_rows.append(dict(
                    classifier=clf_name, dataset=ds_name,
                    enroll_day=enroll_day, test_day=test_day,
                    n_subjects=len(set(y_test)), **res,
                ))

    os.makedirs(OUT_ROOT, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_ROOT, "cross_session_summary.csv"), index=False)
    print("\n\nSaved:", os.path.join(OUT_ROOT, "cross_session_summary.csv"))
    print(summary_df.to_string(index=False))
    print("\nDone.")
