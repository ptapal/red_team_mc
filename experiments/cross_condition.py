import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

warnings.filterwarnings("ignore")

from utils import trial_to_cov_vec, cosine_sim
from arms import AttackArms
from ucb import UCBSelector
from config import SEED, N_ARMS, ARM_NAMES, DEFAULT_C_UCB
from datasets import get_datasets

S_FREQ = 250
LAMBDA = 0.3


CANON_31 = [
    "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3",
    "CP1", "P7", "P3", "Pz", "PO3", "O1", "Oz", "O2",
    "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8", "FC6",
    "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz",
]
CANON_LOWER = [c.lower() for c in CANON_31]

STREAM_BAND = {
    "zhang": (1.0, 10.0), "won": (1.0, 10.0),
    "tanner_eo": (1.0, 30.0), "tanner_ec": (1.0, 30.0),
    "cogbci_eo": (1.0, 30.0), "cogbci_ec": (1.0, 30.0),
    "cho": (1.0, 30.0), "lee": (1.0, 30.0),
}


STREAM_FAMILY = {
    "zhang": "zhang", "won": "won",
    "tanner_eo": "wang", "tanner_ec": "wang",
    "cogbci_eo": "cogbci", "cogbci_ec": "cogbci",
    "cho": "cho", "lee": "lee",
}

PARADIGM = {
    "zhang": "RSVP", "won": "RSVP",
    "tanner_eo": "Resting", "tanner_ec": "Resting",
    "cogbci_eo": "Resting", "cogbci_ec": "Resting",
    "cho": "MI", "lee": "MI",
}


def _pick_canon(df):
    """Reorder a trial CSV onto CANON_31. Returns None if any channel is absent."""
    lut = {c.lower(): c for c in df.index}
    try:
        return df.loc[[lut[c] for c in CANON_LOWER]].values
    except KeyError:
        return None


def load_stream(name, cfg, cap, rng):
    """Load up to `cap` trials per (participant, day), harmonised to CANON_31."""
    sessions = []
    for participant in cfg["participants"]:
        for day in cfg["days"]:
            root = os.path.join(cfg["erp_root"], participant, day)
            if not os.path.isdir(root):
                continue
            files = []
            for block in sorted(os.listdir(root)):
                bpath = os.path.join(root, block)
                if not os.path.isdir(bpath):
                    continue
                files += [os.path.join(bpath, f)
                          for f in sorted(os.listdir(bpath)) if f.endswith(".csv")]
            if len(files) < 4:
                continue
            if len(files) > cap: # even spread, not just the head
                idx = np.linspace(0, len(files) - 1, cap).astype(int)
                files = [files[i] for i in idx]
            trials = []
            for f in files:
                arr = _pick_canon(pd.read_csv(f, index_col=0))
                if arr is not None:
                    trials.append(arr.astype(np.float32)) # upcast at covariance time
            if len(trials) >= 4:
                sessions.append(dict(stream=name, participant=participant,
                                     day=day, X=np.stack(trials)))
    return sessions


def cache_path(scratch, name, cap):
    return os.path.join(scratch, f"cc_{name}_cap{cap}.npz")


def load_stream_cached(name, cfg, cap, rng, scratch):
    p = cache_path(scratch, name, cap)
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        return list(z["sessions"])
    s = load_stream(name, cfg, cap, rng)
    np.savez_compressed(p, sessions=np.array(s, dtype=object))
    return s


def bandpass(X, lo, hi, fs=S_FREQ):
    nyq = fs / 2.0
    hi = min(hi, nyq - 1.0)
    sos = butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, X, axis=-1)


def median_scale(trials):
    return float(np.median([np.median(x.std(axis=1)) for x in trials]))


def prepare_donors(donor_trials, donor_band, enrol_band, t_target, scale_target,
                   rescale=True):
    need_filter = (donor_band[1] > enrol_band[1] + 1e-9) or \
                  (donor_band[0] < enrol_band[0] - 1e-9)
    src_scale = median_scale(donor_trials)
    factor = (scale_target / src_scale) if (rescale and src_scale > 0) else 1.0

    out = []
    for x in donor_trials:
        y = bandpass(x, *enrol_band) if need_filter else x
        if y.shape[1] > t_target:
            y = y[:, :t_target]
        out.append(y * factor)
    return out, need_filter

def build_templates(cov, y):
    t = {}
    for c in np.unique(y):
        m = cov[y == c].mean(axis=0)
        t[int(c)] = m / (np.linalg.norm(m) + 1e-9)
    return t


def compute_tau(templates, cov, y, p):
    genuine = [cosine_sim(v, templates[int(c)])
               for c in np.unique(y) for v in cov[y == c]]
    return float(np.percentile(genuine, p))

def attack(templates, tau, donors, donor_ids, blocked, n_iter, rng, lam=LAMBDA):
    classes = list(templates.keys())
    ucb = UCBSelector(n_arms=N_ARMS, c=DEFAULT_C_UCB)
    arms = AttackArms(rng=rng, s_freq=S_FREQ)
    accepted_flags, per_arm = [], {i: [] for i in range(N_ARMS)}
    per_tgt = {int(c): [] for c in classes}

    for _ in range(n_iter):
        arm = ucb.select()
        di = int(rng.integers(len(donors)))
        X0 = donors[di]

        # a donor that is itself enrolled may not target its own identity
        own = blocked.get(donor_ids[di])
        avail = [c for c in classes if c != own]
        if not avail:
            ucb.update(arm, 0.0)
            continue
        tgt = int(rng.choice(avail))

        imp = None
        if arm == 4: # interpolation partner also from donor pool
            imp = donors[int(rng.integers(len(donors)))]
            if imp.shape != X0.shape:
                imp = None
        try:
            Xp, _ = arms.apply(arm, X0, imp)
        except Exception:
            ucb.update(arm, 0.0)
            continue

        q = trial_to_cov_vec(Xp)
        if not np.all(np.isfinite(q)):
            ucb.update(arm, 0.0)
            continue

        acc = cosine_sim(q, templates[tgt]) >= tau
        delta = float(np.linalg.norm(Xp - X0) / (np.linalg.norm(X0) + 1e-9))
        ucb.update(arm, (1 - lam) * float(acc) + lam * max(0.0, 1.0 - delta))
        accepted_flags.append(bool(acc))
        per_arm[arm].append(bool(acc))
        per_tgt[tgt].append(bool(acc))

    # aggregate per target identity, matching the mean+-std across subjects
    # reported in the within-paradigm results
    tgt_far = [float(np.mean(v)) for v in per_tgt.values() if v]
    arm_far = {ARM_NAMES[i]: (float(np.mean(v)) if v else 0.0) for i, v in per_arm.items()}
    best = max(arm_far, key=arm_far.get) if accepted_flags else None
    return tgt_far, arm_far, best, ucb


def zero_effort(templates, tau, donors, donor_ids, blocked, rng, n_draw=1000):
    """Unmodified donor trials scored against a non-matching enrolled template."""
    classes = list(templates.keys())
    per_tgt = {int(c): [] for c in classes}
    for _ in range(n_draw):
        di = int(rng.integers(len(donors)))
        own = blocked.get(donor_ids[di])
        avail = [c for c in classes if c != own]
        if not avail:
            continue
        q = trial_to_cov_vec(donors[di])
        if not np.all(np.isfinite(q)):
            continue
        tgt = int(rng.choice(avail))
        per_tgt[tgt].append(cosine_sim(q, templates[tgt]) >= tau)
    return [float(np.mean(v)) for v in per_tgt.values() if v]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=100, help="max trials per session")
    ap.add_argument("--iters", type=int, default=1000, help="UCB iterations per enrolled day")
    ap.add_argument("--p", type=int, default=10, help="threshold percentile")
    ap.add_argument("--enrol-frac", type=float, default=0.5)
    ap.add_argument("--no-rescale", action="store_true", help="skip donor amplitude matching")
    ap.add_argument("--symmetric-filter", action="store_true",
                    help="control: when a pair needs band-narrowing, apply the identical "
                         "filter to the enrolled trials too, so both sides have had the "
                         "same number of zero-phase passes")
    ap.add_argument("--streams", type=str, default="", help="comma-separated subset")
    ap.add_argument("--donor-cap", type=int, default=0,
                    help="subsample each donor pool to this many trials (0 = all)")
    ap.add_argument("--out", type=str, default="results/cross_condition")
    ap.add_argument("--scratch", type=str, default=".cc_cache")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.scratch, exist_ok=True)

    cfgs = get_datasets("results")
    names = [s for s in args.streams.split(",") if s] or list(cfgs.keys())
    rng = np.random.default_rng(SEED)

    print(f"cap={args.cap}  iters={args.iters}  p={args.p}  "
          f"rescale={not args.no_rescale}\n")

    store = {}
    for n in names:
        sess = load_stream_cached(n, cfgs[n], args.cap, rng, args.scratch)
        if not sess:
            print(f"  {n}: no sessions, skipping")
            continue
        parts = sorted({s["participant"] for s in sess})
        cls2idx = {p: i for i, p in enumerate(parts)}
        by_day = {}
        test_pool, test_ids = [], []
        for s in sess:
            X = s["X"]
            perm = rng.permutation(len(X))
            cut = max(2, int(len(X) * args.enrol_frac))
            by_day.setdefault(s["day"], []).append(
                dict(cls=cls2idx[s["participant"]],
                     X_enrol=[X[i] for i in perm[:cut]]))
            for i in perm[cut:]:
                test_pool.append(X[i])
                test_ids.append((STREAM_FAMILY[n], s["participant"]))
        if args.donor_cap and len(test_pool) > args.donor_cap:
            keep = rng.choice(len(test_pool), args.donor_cap, replace=False)
            test_pool = [test_pool[i] for i in keep]
            test_ids = [test_ids[i] for i in keep]
        store[n] = dict(
            by_day=by_day, cls2idx=cls2idx, test=test_pool, test_ids=test_ids,
            t_len=sess[0]["X"].shape[2], band=STREAM_BAND[n],
            scale=median_scale([s["X"][0] for s in sess]),
            blocked={(STREAM_FAMILY[n], p): cls2idx[p] for p in parts},
        )
        print(f"  {n:11s} sessions={len(sess):4d}  subjects={len(parts):3d}  "
              f"T={store[n]['t_len']:5d}  scale={store[n]['scale']:.4g}  "
              f"test={len(test_pool)}")

    live = list(store.keys())
    rows = []

    for P in live:
        sp = store[P]
        for Q in live:
            sq = store[Q]
            donors, filtered = prepare_donors(
                sq["test"], sq["band"], sp["band"], sp["t_len"], sp["scale"],
                rescale=not args.no_rescale)
            subj_far, subj_zero, arm_acc = [], [], {n: [] for n in ARM_NAMES}

            for day, day_splits in sorted(sp["by_day"].items()):
                enrol_X = [x for d in day_splits for x in d["X_enrol"]]
                enrol_y = np.array([d["cls"] for d in day_splits
                                    for _ in d["X_enrol"]], dtype=int)
                if args.symmetric_filter and filtered:
                    enrol_X = [bandpass(x, *sp["band"]) for x in enrol_X]
                cov = np.array([trial_to_cov_vec(x) for x in enrol_X])
                templates = build_templates(cov, enrol_y)
                tau = compute_tau(templates, cov, enrol_y, args.p)

                tgt_far, arm_far, best, _ = attack(
                    templates, tau, donors, sq["test_ids"], sp["blocked"],
                    args.iters, rng)
                z = zero_effort(templates, tau, donors, sq["test_ids"],
                                sp["blocked"], rng)
                subj_far += tgt_far
                subj_zero += z
                for n_, v in arm_far.items():
                    arm_acc[n_].append(v)

            far_m = float(np.mean(subj_far)) if subj_far else 0.0
            far_s = float(np.std(subj_far)) if subj_far else 0.0
            zero_m = float(np.mean(subj_zero)) if subj_zero else 0.0
            arm_mean = {n_: float(np.mean(v)) if v else 0.0 for n_, v in arm_acc.items()}
            best_arm = max(arm_mean, key=arm_mean.get)

            rows.append(dict(
                enrolled=P, donor=Q,
                enrolled_paradigm=PARADIGM[P], donor_paradigm=PARADIGM[Q],
                same_stream=int(P == Q),
                same_paradigm=int(PARADIGM[P] == PARADIGM[Q]),
                adv_far=far_m, adv_far_std=far_s, zero_far=zero_m,
                lift=far_m - zero_m, best_arm=best_arm,
                band_matched=int(filtered),
                donor_band_narrower=int(sq["band"][1] < sp["band"][1]),
                **{f"arm_{n_}": v for n_, v in arm_mean.items()},
            ))
            print(f"  enrol {P:11s} <- donor {Q:11s}  "
                  f"adv={far_m:.3f}+-{far_s:.3f}  zero={zero_m:.3f}  "
                  f"lift={far_m - zero_m:+.3f}  best={best_arm}")

    df = pd.DataFrame(rows)
    tag = ("_sym" if args.symmetric_filter else "") + \
          ("_norescale" if args.no_rescale else "")
    df.to_csv(os.path.join(args.out, f"cross_condition_pairs{tag}.csv"), index=False)
    df.pivot(index="enrolled", columns="donor", values="adv_far").to_csv(
        os.path.join(args.out, f"cross_condition_matrix{tag}.csv"))
    df.pivot(index="enrolled", columns="donor", values="zero_far").to_csv(
        os.path.join(args.out, f"cross_condition_zero{tag}.csv"))

    print("\n=== adversarial FAR (rows enrolled, cols donor) ===")
    print(df.pivot(index="enrolled", columns="donor",
                   values="adv_far").round(3).to_string())
    print("\n=== paradigm-level means ===")
    print(df.groupby(["enrolled_paradigm", "donor_paradigm"])[
        ["adv_far", "zero_far", "lift"]].mean().round(3).to_string())
    print(f"\nWrote {args.out}/cross_condition_pairs{tag}.csv")


if __name__ == "__main__":
    main()
