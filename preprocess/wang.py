"""
Preprocess Wang et al. 2022 resting-state EEG (BIDS, ds004148) into the same CSV format used by load_sessions() in helpers.py.

Output directories: <RTMC_DATA>/preprocessed/wang/eo/  — eyes-open condition
                    <RTMC_DATA>/preprocessed/wang/ec/  — eyes-closed condition

Structure mirrors the zhang and won outputs: {out_dir}/{sub_id}/{session}/block/trial_{NNN:03d}.csv
"""

import os
import numpy as np
import pandas as pd
import mne

mne.set_log_level("WARNING")

from paths import RAW_ROOT, PREP_ROOT
from config import (SFREQ_TARGET, TRIAL_SAMPLES, MIN_WINDOWS,
                    WON_CHANNELS_31, BANDPASS_WIDE)

BIDS_ROOT  = str(RAW_ROOT / "wang/ds004148")
OUT_EO     = str(PREP_ROOT / "wang/eo")
OUT_EC     = str(PREP_ROOT / "wang/ec")

SFREQ_OUT  = SFREQ_TARGET
L_FREQ, H_FREQ = BANDPASS_WIDE   # 1-30 Hz: resting state
WIN_SAMPS  = TRIAL_SAMPLES
MIN_WINS   = MIN_WINDOWS

WON_CHANNELS = WON_CHANNELS_31
# lowercase set for matching
WON_LOWER = {c.lower(): c for c in WON_CHANNELS}

SUBJECTS   = [f"sub-{i:02d}" for i in range(1, 61)]
SESSIONS   = ["session1", "session2", "session3"]

TASKS = {
    "eo": "eyesopen",
    "ec": "eyesclosed",
}

def pick_won_channels(raw):
    """Return channel picks for the 32 Won channels (case-insensitive)."""
    ch_names_lower = {c.lower(): c for c in raw.ch_names}
    picks = []
    matched_names = []
    for won_lower, won_orig in WON_LOWER.items():
        if won_lower in ch_names_lower:
            picks.append(ch_names_lower[won_lower])
            matched_names.append(won_orig)   # use Won's capitalization for output
    return picks, matched_names


def process_file(vhdr_path):
    """
    Load a BrainVision file, filter, resample, pick channels.
    Returns (data, ch_names) where data is (32, n_samples) or None on failure.
    """
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)

    ch_picks, ch_names_out = pick_won_channels(raw)
    if len(ch_picks) < len(WON_CHANNELS):
        missing = set(WON_LOWER.keys()) - {c.lower() for c in raw.ch_names}
        print(f"  WARNING: missing {len(missing)} channels in {vhdr_path}: {missing}")

    raw.pick_channels(ch_picks)

    # bandpass 1-30 Hz, zero-phase (MNE default for filter_data is linear phase FIR,
    # which is zero-phase when applied forward-only; use method='iir' to match ERP pipeline)
    raw.filter(L_FREQ, H_FREQ, method="iir", iir_params={"order": 4, "ftype": "butter"},
               verbose=False)

    # resample to 250 Hz
    if raw.info["sfreq"] != SFREQ_OUT:
        raw.resample(SFREQ_OUT, verbose=False)

    data = raw.get_data()  # (32, n_samples) in volts; magnitude doesn't matter for covariance
    return data, ch_names_out


def segment_and_save(data, ch_names, out_dir):
    """
    Segment (32, n_samples) into non-overlapping WIN_SAMPS windows.
    Save each as a CSV trial.
    """
    n_ch, n_samp = data.shape
    n_wins = n_samp // WIN_SAMPS

    if n_wins < MIN_WINS:
        return 0

    os.makedirs(out_dir, exist_ok=True)

    for i in range(n_wins):
        win = data[:, i * WIN_SAMPS:(i + 1) * WIN_SAMPS]   # (32, 162)
        df = pd.DataFrame(win, index=ch_names,
                          columns=np.arange(WIN_SAMPS))
        df.to_csv(os.path.join(out_dir, f"trial_{i+1:03d}.csv"))

    return n_wins



def main():
    for cond, task_name in TASKS.items():
        out_root = OUT_EO if cond == "eo" else OUT_EC
        total_wins = 0
        skipped    = 0

        print(f"\n=== Condition: {cond} ({task_name}) -> {out_root} ===")

        for sub in SUBJECTS:
            for ses in SESSIONS:
                vhdr = os.path.join(
                    BIDS_ROOT, sub, f"ses-{ses}", "eeg",
                    f"{sub}_ses-{ses}_task-{task_name}_eeg.vhdr"
                )
                if not os.path.exists(vhdr):
                    skipped += 1
                    continue

                try:
                    data, ch_names = process_file(vhdr)
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
