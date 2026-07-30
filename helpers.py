import os
import numpy as np
import pandas as pd
from config import MIN_TRIALS_SESSION, ENROL_FRAC

def load_trials(block_path):
    trial_files = sorted([f for f in os.listdir(block_path) if f.endswith(".csv")])
    all_trials = []
    for f in trial_files:
        df = pd.read_csv(os.path.join(block_path, f), index_col=0)
        all_trials.append(df.values)
    return np.stack(all_trials), df.index.tolist()

def load_sessions(participants: list[str],
                  days: list[str],
                  erp_root: str = "ERPs") -> list[dict]:
    sessions = []
    for participant in participants:
        for day in days:
            erp_root_path = f"{erp_root}/{participant}/{day}"
            if not os.path.exists(erp_root_path):
                continue
            blocks = sorted([b for b in os.listdir(erp_root_path)
                              if os.path.isdir(os.path.join(erp_root_path, b))])
            X_sess = []
            for block in blocks:
                trials, _ = load_trials(os.path.join(erp_root_path, block))
                X_sess.extend(trials)
            if len(X_sess) >= MIN_TRIALS_SESSION:
                sessions.append({
                    "participant": participant,
                    "day":         day,
                    "X":           np.stack(X_sess),
                })
    return sessions


def build_within_session_splits(sessions: list[dict],
                                cls2idx: dict[str, int],
                                rng: np.random.Generator,
                                enrol_frac: float = ENROL_FRAC) -> dict[str, dict]:
    """
    Split each subject's trials into enrolment/test within the same day.
    No trial from one day appears in the enrolment or test set of another day.
    """
    by_day: dict[str, list[dict]] = {}
    for sess in sessions:
        by_day.setdefault(sess["day"], []).append(sess)

    day_data = {}
    for day, day_sessions in by_day.items():
        X_enrol, y_enrol = [], []
        X_test,  y_test  = [], []

        for sess in day_sessions:
            cls  = cls2idx[sess["participant"]]
            X_s  = sess["X"]
            n    = len(X_s)
            perm = rng.permutation(n)
            cut  = max(2, int(n * enrol_frac))   # at least 2 enrolment trials

            for i in perm[:cut]:
                X_enrol.append(X_s[i])
                y_enrol.append(cls)
            for i in perm[cut:]:
                X_test.append(X_s[i])
                y_test.append(cls)

        if len(X_enrol) == 0 or len(X_test) == 0:
            continue

        day_data[day] = {
            "X_enrol": X_enrol,
            "y_enrol": np.array(y_enrol, dtype=int),
            "X_test":  X_test,
            "y_test":  np.array(y_test,  dtype=int),
        }

    return day_data
