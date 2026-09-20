"""
Preprocess COG-BCI resting-state EEG (Hinss et al. 2023) into the same CSV format used by load_sessions() in helpers.py.

Participants 1–9 are excluded: electrode Cz was not recorded for them.

Channel note: CP5 is absent in sub-17, sub-25, sub-26, sub-28, sub-29. CP5 is therefore dropped globally.
Output directories: <RTMC_DATA>/preprocessed/cogbci/eo/  — eyes-open condition
                    <RTMC_DATA>/preprocessed/cogbci/ec/  — eyes-closed condition

Structure: {out_dir}/{sub_id}/{session}/block/trial_{NNN:03d}.csv
"""

import os
import numpy as np
import pandas as pd
import mne

mne.set_log_level("WARNING")

from paths import RAW_ROOT, PREP_ROOT
from config import (SFREQ_TARGET, TRIAL_SAMPLES, MIN_WINDOWS,
                    WON_CHANNELS_31, BANDPASS_WIDE)

COGBCI_ROOT = str(RAW_ROOT / "cogbci/cog_bci")
OUT_EO      = str(PREP_ROOT / "cogbci/eo")
OUT_EC      = str(PREP_ROOT / "cogbci/ec")

SFREQ_OUT  = SFREQ_TARGET
L_FREQ, H_FREQ = BANDPASS_WIDE   # 1-30 Hz: resting state
WIN_SAMPS  = TRIAL_SAMPLES
MIN_WINS   = MIN_WINDOWS

WON_CHANNELS = WON_CHANNELS_31
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

        print(f"\n=== Condition: {cond} ({task_file}) -> {out_root} ===")

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
