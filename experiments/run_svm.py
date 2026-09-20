import os
import warnings

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.base import clone

warnings.filterwarnings("ignore")

from helpers  import load_sessions
from utils    import trial_to_cov_vec
from config   import SEED, N_ARMS, ARM_NAMES, DEFAULT_C_UCB
from arms     import AttackArms
from ucb      import UCBSelector
from datasets import get_datasets
from experiments.bandit_all import make_splits, pool_enrol, cov_array, P_SWEEP, P_ATTACK, N_ITER, LAMBDA

DATASETS = get_datasets("results")
OUT_ROOT = "results/svm"

SVM_KERNEL = "linear"
SVM_C      = 1.0


def build_svm_models(cov_enrol: np.ndarray, y_enrol: np.ndarray) -> dict:
    """One-vs-all SVC (RBF, balanced) per enrolled subject, on log-covariance vectors."""
    models = {}
    for cls in np.unique(y_enrol):
        y_bin = (y_enrol == cls).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        clf = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma="scale",
                  class_weight="balanced", random_state=SEED)
        clf.fit(cov_enrol, y_bin)
        models[int(cls)] = clf
    return models


def svm_score(model: SVC, v: np.ndarray) -> float:
    """Signed distance to the hyperplane; higher = more 'genuine'."""
    classes = list(model.classes_)
    raw = float(model.decision_function(v.reshape(1, -1))[0])
    return raw if classes[1] == 1 else -raw


def oof_genuine_scores(cov_enrol: np.ndarray, y_enrol: np.ndarray) -> dict:
    """
    Out-of-fold genuine decision-function scores per subject, via CV on the
    enrolment set.
    """
    out = {}
    for cls in np.unique(y_enrol):
        cls = int(cls)
        y_bin = (y_enrol == cls).astype(int)
        n_pos, n_neg = int(y_bin.sum()), int((1 - y_bin).sum())
        if n_pos < 2 or n_neg < 2:
            continue
        n_splits = max(2, min(5, n_pos, n_neg))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        base = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma="scale",
                   class_weight="balanced", random_state=SEED)
        oof = cross_val_predict(clone(base), cov_enrol, y_bin, cv=cv, method="decision_function")
        # decision_function sign follows classes_[1]; with two classes present
        # in every fold this is consistently the positive (genuine) class here
        out[cls] = oof[y_bin == 1]
    return out


def compute_tau_svm(oof_scores: dict, p: float) -> float:
    genuine = np.concatenate([v for v in oof_scores.values() if len(v)])
    return float(np.percentile(genuine, p))


def threshold_sweep_svm(sessions: list, out_dir: str, ds_name: str):
    rng = np.random.default_rng(SEED)
    all_participants = sorted({s["participant"] for s in sessions})
    cls2idx = {p: i for i, p in enumerate(all_participants)}
    splits  = make_splits(sessions, rng)
    by_day  = pool_enrol(splits)

    p2frr, p2far = {p: [] for p in P_SWEEP}, {p: [] for p in P_SWEEP}

    for day, day_splits in by_day.items():
        all_enrol_X = [x for sp in day_splits for x in sp["X_enrol"]]
        all_enrol_y = [cls2idx[sp["participant"]] for sp in day_splits for _ in sp["X_enrol"]]
        cov_enrol_all = cov_array(all_enrol_X)
        y_enrol_all   = np.array(all_enrol_y, dtype=int)
        models_all    = build_svm_models(cov_enrol_all, y_enrol_all)
        if not models_all:
            continue
        oof_all = oof_genuine_scores(cov_enrol_all, y_enrol_all)

        for p in P_SWEEP:
            tau = compute_tau_svm(oof_all, p)
            classes = list(models_all.keys())

            for sp in day_splits:
                cls = cls2idx[sp["participant"]]
                if cls not in models_all or not sp["X_test"]:
                    continue
                test_vecs  = cov_array(sp["X_test"])
                gen_scores = [svm_score(models_all[cls], v) for v in test_vecs]
                p2frr[p].append(float(np.mean(np.array(gen_scores) < tau)))

                others = [c for c in classes if c != cls]
                if others:
                    imp_cls    = int(rng.choice(others))
                    imp_scores = [svm_score(models_all[imp_cls], v) for v in test_vecs]
                    p2far[p].append(float(np.mean(np.array(imp_scores) >= tau)))

    results = {p: dict(mean_frr=np.mean(p2frr[p]) if p2frr[p] else 0.0,
                        std_frr=np.std(p2frr[p])   if p2frr[p] else 0.0,
                        mean_far=np.mean(p2far[p]) if p2far[p] else 0.0,
                        std_far=np.std(p2far[p])   if p2far[p] else 0.0)
               for p in P_SWEEP}

    diffs   = [abs(results[p]["mean_frr"] - results[p]["mean_far"]) for p in P_SWEEP]
    eer_p   = P_SWEEP[int(np.argmin(diffs))]
    eer_val = (results[eer_p]["mean_frr"] + results[eer_p]["mean_far"]) / 2

    print(f"{ds_name} (SVM) EER approx {eer_val:.3f} at p={eer_p}")
    rows = [dict(p=p, FRR_mean=results[p]["mean_frr"], FRR_std=results[p]["std_frr"],
                 FAR_mean=results[p]["mean_far"],  FAR_std=results[p]["std_far"])
            for p in P_SWEEP]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "threshold_sweep_std.csv"), index=False)

    return results, eer_p, eer_val


def _attack_loop_svm(models, tau, X_test, y_test, n_iter, rng, lam=LAMBDA):
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
        score    = svm_score(models[tgt], q)
        accepted = score >= tau
        delta    = float(np.linalg.norm(Xp - X0) / (np.linalg.norm(X0) + 1e-9))
        reward   = (1 - lam) * float(accepted) + lam * max(0.0, 1.0 - delta)
        ucb.update(arm, reward)
        logs.append(dict(arm=arm, arm_name=ARM_NAMES[arm],
                         tgt=tgt, accepted=bool(accepted), reward=float(reward)))
        per_tgt[tgt].append(bool(accepted))

    return logs, per_tgt, ucb


def run_svm_attack(sessions: list, out_dir: str, ds_name: str) -> dict:
    rng = np.random.default_rng(SEED)
    all_participants = sorted({s["participant"] for s in sessions})
    cls2idx  = {p: i for i, p in enumerate(all_participants)}
    splits   = make_splits(sessions, rng)
    by_day   = pool_enrol(splits)

    # build once per day, reuse across all P_ATTACK values
    day_cache = {}
    for day, day_splits in sorted(by_day.items()):
        all_enrol_X = [x for sp in day_splits for x in sp["X_enrol"]]
        all_enrol_y = [cls2idx[sp["participant"]] for sp in day_splits for _ in sp["X_enrol"]]
        cov_enrol = cov_array(all_enrol_X)
        y_enrol   = np.array(all_enrol_y, dtype=int)
        models    = build_svm_models(cov_enrol, y_enrol)
        if not models:
            continue
        oof = oof_genuine_scores(cov_enrol, y_enrol)

        all_test_X = [x for sp in day_splits for x in sp["X_test"]]
        all_test_y = [cls2idx[sp["participant"]] for sp in day_splits for _ in sp["X_test"]]
        y_test = np.array(all_test_y, dtype=int)

        day_cache[day] = dict(models=models, oof=oof,
                               X_test=all_test_X, y_test=y_test)

    all_results = {}
    for p in P_ATTACK:
        arm_logs_all = {name: [] for name in ARM_NAMES}
        per_subj_far = []

        for day, dc in sorted(day_cache.items()):
            tau = compute_tau_svm(dc["oof"], p)
            logs, per_tgt, ucb = _attack_loop_svm(
                dc["models"], tau, dc["X_test"], dc["y_test"], N_ITER, rng)
            if not logs:
                continue

            for cls_idx, accepted_list in per_tgt.items():
                if accepted_list:
                    per_subj_far.append(float(np.mean(accepted_list)))

            for name in ARM_NAMES:
                arm_l = [l for l in logs if l["arm_name"] == name]
                if arm_l:
                    arm_logs_all[name].append(float(np.mean([l["accepted"] for l in arm_l])))

        all_results[p] = dict(
            mean_far = float(np.mean(per_subj_far)) if per_subj_far else 0.0,
            std_far  = float(np.std(per_subj_far))  if per_subj_far else 0.0,
            arm={name: (float(np.mean(v)) if v else 0.0,
                        float(np.std(v))  if v else 0.0)
                 for name, v in arm_logs_all.items()},
        )

        r = all_results[p]
        print(f"{ds_name} (SVM) p={p}: adversarial FAR = {r['mean_far']:.3f}+-{r['std_far']:.3f}")
        best = max(r["arm"], key=lambda n: r["arm"][n][0])
        print(f"  Best arm (raw FAR): {best}")

    _plot_arms(all_results, ds_name, out_dir)

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
    r     = adv_results[p_ref]
    means = [r["arm"][n][0] for n in ARM_NAMES]
    stds  = [r["arm"][n][1] for n in ARM_NAMES]
    idx   = np.argsort(means)[::-1]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(ARM_NAMES)), [means[i] for i in idx], color="darkred",
           yerr=[stds[i] for i in idx], capsize=4, alpha=0.85)
    ax.set_xticks(range(len(ARM_NAMES)))
    ax.set_xticklabels([ARM_NAMES[i] for i in idx], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Adversarial FAR", fontsize=12)
    ax.set_title(f"{ds_name} (SVM) -- per-arm adversarial FAR at $p=10$ (+-1 std)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "arm_far_std.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    summary = {}

    for ds_name, cfg in DATASETS.items():
        out_dir = os.path.join(OUT_ROOT, ds_name)
        print(f"\n{'='*64}\n  {ds_name.upper()}  (SVM)\n{'='*64}")
        os.makedirs(out_dir, exist_ok=True)

        sessions = load_sessions(cfg["participants"], cfg["days"], erp_root=cfg["erp_root"])
        print(f"Sessions loaded: {len(sessions)}  "
              f"(subjects: {len({s['participant'] for s in sessions})})")
        if not sessions:
            print("  No sessions -- skipping.")
            continue

        sw, eer_p, eer = threshold_sweep_svm(sessions, out_dir, ds_name)
        adv            = run_svm_attack(sessions, out_dir, ds_name)

        summary[ds_name] = dict(sweep=sw, eer_p=eer_p, eer=eer, adv=adv)

    print("\n\n" + "="*64)
    print("SUMMARY (SVM)")
    print("="*64)
    for ds in DATASETS:
        if ds not in summary:
            continue
        s = summary[ds]
        p10 = s["sweep"][10]
        print(f"\n{ds.upper()}")
        print(f"  EER approx {s['eer']:.3f} at p={s['eer_p']}")
        print(f"  p=10: FRR={p10['mean_frr']:.3f}+-{p10['std_frr']:.3f}  "
              f"FAR={p10['mean_far']:.3f}+-{p10['std_far']:.3f}")
        for p in P_ATTACK:
            r = s["adv"][p]
            best = max(r["arm"], key=lambda n: r["arm"][n][0])
            print(f"  adv p={p:>2}: FAR={r['mean_far']:.3f}+-{r['std_far']:.3f}  best={best}")

    print("\nDone.")
