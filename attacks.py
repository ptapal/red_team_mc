from auth import RiemannAuthenticator
from ucb import UCBSelector
from arms import AttackArms
import numpy as np
from config import N_ARMS, ARM_NAMES, SEED, DEFAULT_C_UCB, DEFAULT_N_ITER, DEFAULT_OUTPUT_DIR, ENROL_FRAC, s_freq
import os
import pandas as pd
from utils import trial_to_cov_vec
from helpers import build_within_session_splits
from plots import _plot_cross_day, _plot_day

def _attack_loop(auth: RiemannAuthenticator,
                 X_test: list,
                 y_test: np.ndarray,
                 ucb: UCBSelector,
                 arms: AttackArms,
                 n_iter: int,
                 day: str,
                 rng: np.random.Generator) -> tuple[list[dict], dict[int, list]]:
    """
    Run n_iter UCB attack iterations against a fitted authenticator.

    X_test  : list of raw EEG trials (attacks need raw signal, not cov vecs)
    y_test  : integer labels aligned with X_test
    Both must come exclusively from the same day as the enrolment data.
    """
    X_test_arr = np.array(X_test)
    classes    = np.unique(y_test)
    logs: list[dict] = []
    strength_logs: dict[int, list] = {i: [] for i in range(N_ARMS)}

    for t in range(n_iter):
        arm = ucb.select()

        # Sample victim trial from within this day's test pool
        victim_i     = int(rng.integers(len(X_test_arr)))
        trial_orig   = X_test_arr[victim_i]
        true_subject = int(y_test[victim_i])

        # Impersonation arm: impostor must also come from this day's test pool
        impostor_trial = None
        if arm == 4:
            other_cls = [c for c in classes if c != true_subject]
            if len(other_cls) > 0:
                imp_cls  = int(rng.choice(other_cls))
                imp_mask = np.where(y_test == imp_cls)[0]
                if len(imp_mask) > 0:
                    impostor_trial = X_test_arr[int(rng.choice(imp_mask))]

        try:
            perturbed, params = arms.apply(arm, trial_orig, impostor_trial)
        except Exception as e:
            print(f"    [warn] day={day} arm={arm} failed: {e}")
            ucb.update(arm, 0.0)
            continue

        # Attacker claims to be a different subject → directly measures FAR
        other_cls      = [c for c in classes if c != true_subject]
        target_subject = int(rng.choice(other_cls)) if len(other_cls) > 0 \
                         else int(true_subject)

        # Guard: skip if target has no template (edge case for small sessions)
        if target_subject not in auth.templates:
            ucb.update(arm, 0.0)
            continue

        cov_vec         = trial_to_cov_vec(perturbed)
        accepted, score = auth.verify(cov_vec, claimed_id=target_subject)

        reward = 1.0 if accepted else 0.0
        ucb.update(arm, reward)

        strength = float(np.linalg.norm(perturbed - trial_orig) /
                         (np.linalg.norm(trial_orig) + 1e-9))
        strength_logs[arm].append((strength, int(accepted)))

        logs.append({
            "day":            day,
            "iter":           t,
            "arm":            arm,
            "arm_name":       ARM_NAMES[arm],
            "target_subject": target_subject,
            "true_subject":   true_subject,
            "score":          round(float(score), 5),
            "accepted":       bool(accepted),
            "strength":       round(strength, 5),
            **{k: v for k, v in params.items() if k not in ("arm", "name")},
        })

        if (t + 1) % 200 == 0:
            far = sum(l["accepted"] for l in logs) / len(logs)
            print(f"      iter {t+1:>5}/{n_iter}  "
                  f"arm={ARM_NAMES[arm]:<22}  cumul FAR={far:.3f}")

    return logs, strength_logs


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_redteam(sessions:    list[dict],
                cls2idx:     dict[str, int],
                n_iter:      int   = DEFAULT_N_ITER,
                c_ucb:       float = DEFAULT_C_UCB,
                output_dir:  str   = DEFAULT_OUTPUT_DIR,
                seed:        int   = SEED) -> dict:
    """
    Within-session red-teaming (Option A).

    One independent UCB bandit is run per day. Enrolment and attack trials
    are drawn exclusively from the same participant x day session —
    no cross-day contamination at any point.

    Parameters
    ----------
    sessions   : output of load_sessions()
    cls2idx    : participant-string to integer-label mapping
    n_iter     : UCB iterations per day
    c_ucb      : UCB exploration constant
    output_dir : root directory for saved results
    seed       : global RNG seed

    Returns
    -------
    dict keyed by day string, each containing logs / FAR tables / UCB state
    """
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    # ── Build within-session splits (no cross-day leakage) ───────────────────
    print("Building within-session enrolment / test splits …")
    day_data = build_within_session_splits(sessions, cls2idx, rng, ENROL_FRAC)

    if not day_data:
        raise RuntimeError("No valid day splits found. Check your data paths.")

    all_results = {}

    # ── Per-day loop ──────────────────────────────────────────────────────────
    for day, data in sorted(day_data.items()):
        n_subjects = len(np.unique(data["y_enrol"]))
        print(f"\n{'='*62}")
        print(f"  Day: {day}  |  enrol={len(data['X_enrol'])}  |  "
              f"test={len(data['X_test'])}  |  subjects={n_subjects}")
        print(f"{'='*62}")

        # 1. Precompute covariance vectors for THIS day's enrolment trials only
        print("  Computing enrolment covariances …")
        cov_enrol = np.array([trial_to_cov_vec(x) for x in data["X_enrol"]])

        # 2. Fit authenticator on THIS day's enrolment data only
        auth = RiemannAuthenticator(threshold_percentile=10.0)
        auth.fit(cov_enrol, data["y_enrol"])
        print(f"  tau = {auth.tau:.4f}  |  "
              f"templates: {sorted(auth.templates.keys())}")

        # 3. Fresh UCB bandit and attack arms for this day
        ucb  = UCBSelector(n_arms=N_ARMS, c=c_ucb)
        arms = AttackArms(rng=rng, s_freq=s_freq)

        # 4. Attack loop — test trials are from this same day only
        print(f"  Running {n_iter} UCB iterations …")
        logs, strength_logs = _attack_loop(
            auth   = auth,
            X_test = data["X_test"],   # same-day test trials
            y_test = data["y_test"],
            ucb    = ucb,
            arms   = arms,
            n_iter = n_iter,
            day    = day,
            rng    = rng,
        )

        if not logs:
            print(f"  [warn] No valid attack logs for {day}, skipping.")
            continue

        # 5. Per-arm FAR
        far_per_arm = {}
        for i, name in enumerate(ARM_NAMES):
            arm_logs = [l for l in logs if l["arm"] == i]
            far_per_arm[name] = (sum(l["accepted"] for l in arm_logs) / len(arm_logs)
                                 if arm_logs else 0.0)

        # 6. FAR vs perturbation strength
        far_vs_strength = {}
        for arm_i, slog in strength_logs.items():
            if not slog:
                continue
            strengths = np.array([s for s, _ in slog])
            successes = np.array([a for _, a in slog])
            bins      = np.linspace(0, strengths.max() + 1e-9, 11)
            bin_far, bin_mid = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (strengths >= lo) & (strengths < hi)
                if mask.sum() > 0:
                    bin_far.append(float(successes[mask].mean()))
                    bin_mid.append(float((lo + hi) / 2))
            far_vs_strength[arm_i] = {"bin_mid": bin_mid, "far": bin_far}

        # 7. Wilcoxon: best vs worst arm for this day
        arm_rewards = {i: [l["accepted"] for l in logs if l["arm"] == i]
                       for i in range(N_ARMS)}
        sorted_arms = sorted(arm_rewards,
                              key=lambda i: np.mean(arm_rewards[i]) if arm_rewards[i] else 0,
                              reverse=True)
        best_arm, worst_arm = sorted_arms[0], sorted_arms[-1]

        from scipy.stats import wilcoxon as _wilcoxon
        best_r  = np.array(arm_rewards[best_arm],  dtype=float)
        worst_r = np.array(arm_rewards[worst_arm], dtype=float)
        if len(best_r) > 1 and len(worst_r) > 1:
            min_len = min(len(best_r), len(worst_r))
            stat, p = _wilcoxon(best_r[:min_len], worst_r[:min_len])
            print(f"\n  Wilcoxon ({ARM_NAMES[best_arm]} vs {ARM_NAMES[worst_arm]}): "
                  f"stat={stat:.3f}, p={p:.4f}")

        # 8. Console summary
        overall_far = sum(l["accepted"] for l in logs) / len(logs)
        print(f"\n  Overall FAR [{day}]: {overall_far:.3f}")
        print(f"  {'Arm':<26}  {'FAR':>6}  {'Pulls':>6}  {'UCB':>8}")
        print(f"  {'─'*52}")
        for i, name in enumerate(ARM_NAMES):
            print(f"  {name:<26}  "
                  f"{far_per_arm[name]:>6.3f}  "
                  f"{ucb.counts[i]:>6}  "
                  f"{ucb.ucb_scores()[i]:>8.4f}")

        # 9. Save per-day artefacts
        day_dir = os.path.join(output_dir, day)
        os.makedirs(day_dir, exist_ok=True)

        pd.DataFrame(logs).to_csv(
            os.path.join(day_dir, "attack_log.csv"), index=False)

        summary = {
            "day":         day,
            "n_iter":      n_iter,
            "c_ucb":       c_ucb,
            "tau":         auth.tau,
            "overall_FAR": overall_far,
            "far_per_arm": far_per_arm,
            "ucb_counts":  ucb.counts.tolist(),
            "ucb_means":   ucb.means.tolist(),
            "ucb_scores":  ucb.ucb_scores().tolist(),
        }

        _plot_day(ucb, far_per_arm, far_vs_strength, logs, day, day_dir)

        all_results[day] = {
            "ucb":             ucb,
            "logs":            logs,
            "far_per_arm":     far_per_arm,
            "far_vs_strength": far_vs_strength,
            "overall_FAR":     overall_far,
        }

    # Cross-day summary plot (does not use cross-day data, just aggregates results)
    _plot_cross_day(all_results, output_dir)

    return all_results