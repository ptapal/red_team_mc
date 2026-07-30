"""
Preprocess Lee et al. 2019 motor imagery EEG into the same CSV format used by load_sessions() in helpers.py.

Output directory:
  ERPs_lee/

Structure: {out_dir}/{sub_id}/{session}/block/trial_{NNN:03d}.csv
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import os
import numpy as np
import pandas as pd
import scipy.io
import mne

mne.set_log_level("WARNING")

LEE_ROOT = "/Users/a1234/red_team_mc/data/raw/lee/gigadb-datasets/live/pub/10.5524/100001_101000/100542"
OUT_DIR  = "/Users/a1234/red_team_mc/data/preprocessed/lee"

SFREQ_IN  = 1000
SFREQ_OUT = 250
L_FREQ    = 1.0
H_FREQ    = 30.0

N_CH      = 62      # total Lee EEG channels (used to build RawArray)
SAMPS_IN  = 4000    # samples per trial at 1000 Hz (4 s)
SAMPS_OUT = 1000    # samples per trial at 250 Hz  (4 s)

# All 62 Lee channel names
LEE_ALL_CH = [
    "Fp1", "Fp2", "F7",  "F3",  "Fz",    "F4",    "F8",    "FC5",
    "FC1", "FC2", "FC6", "T7",  "C3",    "Cz",    "C4",    "T8",
    "TP9", "CP5", "CP1", "CP2", "CP6",   "TP10",  "P7",    "P3",
    "Pz",  "P4",  "P8",  "PO9", "O1",    "Oz",    "O2",    "PO10",
    "FC3", "FC4", "C5",  "C1",  "C2",    "C6",    "CP3",   "CPz",
    "CP4", "P1",  "P2",  "POz", "FT9",   "FTT9h", "TTP7h", "TP7",
    "TPP9h","FT10","FTT10h","TPP8h","TP8","TPP10h","F9",   "F10",
    "AF7", "AF3", "AF4", "AF8", "PO3",   "PO4",
]

# 32-channel Won subset ; matches RSVP pair (C=32, D=528)
WON_CHANNELS = [
    "Fp1", "AF3", "F7",  "F3",  "FC1", "FC5", "T7",  "C3",
    "CP1", "CP5", "P7",  "P3",  "Pz",  "PO3", "O1",  "Oz",
    "O2",  "PO4", "P4",  "P8",  "CP6", "CP2", "C4",  "T8",
    "FC6", "FC2", "F4",  "F8",  "AF4", "Fp2", "Fz",  "Cz",
]

SESSIONS  = ["session1", "session2"]
N_SUBJECTS = 54
MIN_TRIALS = 10



def mat_path(session, sub_idx):
    """Return path to Lee mat file. Directory names are not zero-padded."""
    sess_num = session[-1]   # "1" or "2"
    return os.path.join(
        LEE_ROOT, session,
        f"s{sub_idx}",
        f"sess0{sess_num}_subj{sub_idx:02d}_EEG_MI.mat",
    )


def load_and_process(path):
    """
    Load one session mat file and return (n_ch, n_trials, samps_out).
    Train and test splits are pooled. Returns None on failure.
    """
    mat = scipy.io.loadmat(path, simplify_cells=True)

    splits = []
    for key in ("EEG_MI_train", "EEG_MI_test"):
        smt = np.array(mat[key]["smt"], dtype=np.float64)  # (4000, 100, 62)
        # transpose -> (62, 100, 4000)
        splits.append(smt.transpose(2, 1, 0))

    # pool -> (62, 200, 4000)
    pooled = np.concatenate(splits, axis=1)
    n_trials = pooled.shape[1]

    # concatenate trials end-to-end for single-pass filtering -> (62, 200*4000)
    flat = pooled.reshape(N_CH, n_trials * SAMPS_IN)

    info = mne.create_info(ch_names=LEE_ALL_CH, sfreq=SFREQ_IN, ch_types="eeg")
    raw  = mne.io.RawArray(flat, info, verbose=False)
    raw.pick_channels(WON_CHANNELS, ordered=True)
    raw.filter(L_FREQ, H_FREQ, method="iir",
               iir_params={"order": 4, "ftype": "butter"}, verbose=False)
    raw.resample(SFREQ_OUT, verbose=False)
    data_filt = raw.get_data()   # (32, n_trials * SAMPS_OUT)

    n_out = len(WON_CHANNELS)
    return data_filt.reshape(n_out, n_trials, SAMPS_OUT)



def main():
    total_trials = 0
    skipped      = 0

    for sub_idx in range(1, N_SUBJECTS + 1):
        sub_id = f"sub-{sub_idx:02d}"

        for session in SESSIONS:
            path = mat_path(session, sub_idx)
            if not os.path.exists(path):
                skipped += 1
                continue

            try:
                epochs = load_and_process(path)   # (32, 200, 1000)
            except Exception as e:
                print(f"  ERROR {sub_id}/{session}: {e}")
                skipped += 1
                continue

            n_trials = epochs.shape[1]
            if n_trials < MIN_TRIALS:
                print(f"  SKIP {sub_id}/{session}: only {n_trials} trials")
                skipped += 1
                continue

            out_block = os.path.join(OUT_DIR, sub_id, session, "block")
            os.makedirs(out_block, exist_ok=True)

            for i in range(n_trials):
                trial = epochs[:, i, :]   # (32, 1000)
                df = pd.DataFrame(trial, index=WON_CHANNELS,
                                  columns=np.arange(SAMPS_OUT))
                df.to_csv(os.path.join(out_block, f"trial_{i+1:03d}.csv"))

            total_trials += n_trials
            print(f"  {sub_id}/{session}: {n_trials} trials saved")

    print(f"\nDone. Total trials: {total_trials}, files skipped: {skipped}")


if __name__ == "__main__":
    main()
