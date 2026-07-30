"""
Preprocess COG-BCI resting-state EEG (Hinss et al. 2023) into the same
CSV format used by load_sessions() in helpers.py.

We use the session-opening resting-state block only (RS_Beg_EO / RS_Beg_EC),
one minute each, yielding ~92 trials per condition per session.

Participants 1–9 are excluded: electrode Cz was not recorded for them.
Remaining: sub-10 through sub-29 → 20 complete participants.

Channel note: CP5 is absent in sub-17, sub-25, sub-26, sub-28, sub-29.
CP5 is therefore dropped globally so all 20 subjects use the same 31-channel
set (C=31, feature dim D=31×32/2=496).

Output directories:
  ERPs_cogbci_eo/  — eyes-open condition
  ERPs_cogbci_ec/  — eyes-closed condition

Structure mirrors existing ERPs dirs:
  {out_dir}/{sub_id}/{session}/block/trial_{NNN:03d}.csv

Preprocessing:
  - EEGLAB .set → MNE RawEEGLAB
  - Drop ECG1 channel
  - Pick 31 channels (Won set minus CP5)
  - Bandpass 1–30 Hz (IIR zero-phase; covers alpha 8–13 Hz and beta 13–30 Hz)
  - Resample 500 → 250 Hz
  - Non-overlapping 162-sample (648 ms) windows, no baseline correction
"""

import os
import numpy as np
import pandas as pd
import mne

mne.set_log_level("WARNING")

COGBCI_ROOT = "/Users/a1234/red_team_mc/data/raw/cogbci/cog_bci"
OUT_EO      = "/Users/a1234/red_team_mc/data/preprocessed/cogbci/eo"
OUT_EC      = "/Users/a1234/red_team_mc/data/preprocessed/cogbci/ec"

SFREQ_OUT  = 250
L_FREQ     = 1.0
H_FREQ     = 30.0
WIN_SAMPS  = 162
MIN_WINS   = 5

# CP5 absent in sub-17/25/26/28/29 → dropped globally for consistency (31 channels)
WON_CHANNELS = [
    "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7",  "C3",
    "CP1",        "P7", "P3", "Pz",  "PO3", "O1",  "Oz",
    "O2",  "PO4", "P4", "P8", "CP6", "CP2", "C4",  "T8",
    "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz",  "Cz",
]
WON_LOWER = {c.lower(): c for c in WON_CHANNELS}

# sub-01 to sub-09 excluded (no Cz); sub-10 to sub-29 retained
SUBJECTS  = [f"sub-{i:02d}" for i in range(10, 30)]
SESSIONS  = ["S1", "S2", "S3"]

TASKS = {
    "eo": "RS_Beg_EO",
    "ec": "RS_Beg_EC",
}


def pick_won_channels(raw):
    ch_lower = {c.lower(): c for c in raw.ch_names}
    picks, names_out = [], []
    for won_lower, won_orig in WON_LOWER.items():
        if won_lower in ch_lower:
            picks.append(ch_lower[won_lower])
            names_out.append(won_orig)
    return picks, names_out


def process_file(set_path):
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)

    # drop ECG
    if "ECG1" in raw.ch_names:
        raw.drop_channels(["ECG1"])

    ch_picks, ch_names_out = pick_won_channels(raw)
    if len(ch_picks) < len(WON_CHANNELS):
        missing = set(WON_LOWER.keys()) - {c.lower() for c in raw.ch_names}
        print(f"  WARNING: missing channels {missing} in {set_path}")

    raw.pick_channels(ch_picks)
    raw.filter(L_FREQ, H_FREQ, method="iir",
               iir_params={"order": 4, "ftype": "butter"}, verbose=False)
    if raw.info["sfreq"] != SFREQ_OUT:
        raw.resample(SFREQ_OUT, verbose=False)

    return raw.get_data(), ch_names_out


def segment_and_save(data, ch_names, out_dir):
    n_wins = data.shape[1] // WIN_SAMPS
    if n_wins < MIN_WINS:
        return 0
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_wins):
        win = data[:, i * WIN_SAMPS:(i + 1) * WIN_SAMPS]
        pd.DataFrame(win, index=ch_names,
                     columns=np.arange(WIN_SAMPS)).to_csv(
            os.path.join(out_dir, f"trial_{i+1:03d}.csv"))
    return n_wins


def main():
    for cond, task_file in TASKS.items():
        out_root   = OUT_EO if cond == "eo" else OUT_EC
        total_wins = 0
        skipped    = 0

        print(f"\n=== Condition: {cond} ({task_file}) → {out_root} ===")

        for sub in SUBJECTS:
            for ses in SESSIONS:
                set_path = os.path.join(
                    COGBCI_ROOT, sub, f"ses-{ses}", "eeg", f"{task_file}.set"
                )
                if not os.path.exists(set_path):
                    print(f"  MISSING: {sub}/ses-{ses}")
                    skipped += 1
                    continue

                try:
                    data, ch_names = process_file(set_path)
                except Exception as e:
                    print(f"  ERROR {sub}/{ses}: {e}")
                    skipped += 1
                    continue

                out_block = os.path.join(out_root, sub, ses, "block")
                n = segment_and_save(data, ch_names, out_block)
                if n < MIN_WINS:
                    print(f"  SKIP {sub}/{ses}: only {n} windows")
                    skipped += 1
                else:
                    total_wins += n

        print(f"Done. Total windows: {total_wins}, skipped: {skipped}")


if __name__ == "__main__":
    main()
