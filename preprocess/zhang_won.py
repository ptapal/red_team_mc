"""
Won et al. (2022) RSVP preprocessing pipeline.

Outputs: <RTMC_DATA>/preprocessed/won/S{sub_num}/Day_1/block_1/trial_{n}.csv   — Won trials
         <RTMC_DATA>/preprocessed/zhang/won_channels/{participant}/{day}/{block}/trial_N.csv — Zhang 32-ch
"""

import os
import numpy as np
import pandas as pd
import mne

mne.set_log_level("ERROR")

from paths import RAW_ROOT, PREP_ROOT
from config import SFREQ_TARGET as SFREQ_CFG, TRIAL_SAMPLES, BANDPASS_RSVP

BIDS_ROOT          = str(RAW_ROOT / "won/eeg_bids/Won2022_BIDS")
WON_OUT            = str(PREP_ROOT / "won")
ZHANG_ROOT         = str(PREP_ROOT / "zhang/full_epochs")
ZHANG_REDUCED_OUT  = str(PREP_ROOT / "zhang/won_channels")

SFREQ_TARGET = SFREQ_CFG    # Hz; match Zhang
L_FREQ, H_FREQ = BANDPASS_RSVP   # 1-10 Hz: RSVP, narrower than the other paradigms
T_PRE        = 0.050        # 50 ms pre-stimulus
N_SAMPLES    = TRIAL_SAMPLES  # must equal Zhang trial length
# tmax computed so that after resample we get exactly N_SAMPLES
T_POST = N_SAMPLES / SFREQ_TARGET - T_PRE   # 0.598 s

MIN_TRIALS = 4

CHAN_MAP = {
    "FP1": "Fp1",  "AF3": "Af3",  "F7":  "F7",   "F3":  "F3",
    "FC1": "Fc1",  "FC5": "Fc5",  "T7":  "T7",   "C3":  "C3",
    "CP1": "Cp1",  "CP5": "Cp5",  "P7":  "P7",   "P3":  "P3",
    "Pz":  "Pz",   "PO3": "Po3",  "O1":  "O1",   "Oz":  "Oz",
    "O2":  "O2",   "PO4": "Po4",  "P4":  "P4",   "P8":  "P8",
    "CP6": "Cp6",  "CP2": "Cp2",  "C4":  "C4",   "T8":  "T8",
    "FC6": "Fc6",  "FC2": "Fc2",  "F4":  "F4",   "F8":  "F8",
    "AF4": "Af4",  "FP2": "Fp2",  "FZ":  "Fz",   "Cz":  "Cz",
}

# 32 common channel names in Zhang convention (preserved in Won's Biosemi order)
COMMON_CHANS = [CHAN_MAP[ch] for ch in [
    "FP1","AF3","F7","F3","FC1","FC5","T7","C3",
    "CP1","CP5","P7","P3","Pz","PO3","O1","Oz",
    "O2","PO4","P4","P8","CP6","CP2","C4","T8",
    "FC6","FC2","F4","F8","AF4","FP2","FZ","Cz",
]]

TIME_COLS = [str(i) for i in range(N_SAMPLES)]


def process_won_subject(sub_dir: str, sub_label: str, out_root: str) -> int:
    sub_id = os.path.basename(sub_dir)
    set_file = os.path.join(sub_dir, "eeg",
                            f"{sub_id}_task-RSVPtask_run-3_eeg.set")
    if not os.path.exists(set_file):
        return 0

    raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)

    # bandpass 1-10 Hz (IIR, same band as Zhang et al.)
    raw.filter(L_FREQ, H_FREQ, method="iir", verbose=False)

    events, _ = mne.events_from_annotations(raw, verbose=False)
    target_events = events[events[:, 2] == 1]

    if len(target_events) < MIN_TRIALS:
        print(f"  {sub_label}: only {len(target_events)} targets — skip")
        return 0

    epochs = mne.Epochs(
        raw, target_events, event_id=1,
        tmin=-T_PRE, tmax=T_POST,
        baseline=(-T_PRE, 0.0),
        preload=True, verbose=False,
        event_repeated="drop",
    )
    epochs.resample(SFREQ_TARGET, verbose=False)

    data = epochs.get_data() * 1e6    # convert V -> uV to match Zhang units
    n_times = data.shape[2]
    if n_times > N_SAMPLES:
        data = data[:, :, :N_SAMPLES]
    elif n_times < N_SAMPLES:
        pad = np.zeros((data.shape[0], data.shape[1], N_SAMPLES - n_times))
        data = np.concatenate([data, pad], axis=2)

    out_block = os.path.join(out_root, sub_label, "Day_1", "block_1")
    os.makedirs(out_block, exist_ok=True)

    for i, trial in enumerate(data):
        df = pd.DataFrame(trial, index=COMMON_CHANS, columns=TIME_COLS)
        df.to_csv(os.path.join(out_block, f"trial_{i+1}.csv"))

    return len(data)


def reduce_zhang_subject(participant: str) -> int:
    src = os.path.join(ZHANG_ROOT, participant)
    n_saved = 0
    for day in sorted(os.listdir(src)):
        day_path = os.path.join(src, day)
        if not os.path.isdir(day_path):
            continue
        for block in sorted(os.listdir(day_path)):
            block_path = os.path.join(day_path, block)
            if not os.path.isdir(block_path):
                continue
            out_block = os.path.join(ZHANG_REDUCED_OUT, participant, day, block)
            os.makedirs(out_block, exist_ok=True)
            for fname in sorted(os.listdir(block_path)):
                if not fname.endswith(".csv"):
                    continue
                df = pd.read_csv(os.path.join(block_path, fname), index_col=0)
                df.loc[COMMON_CHANS].to_csv(os.path.join(out_block, fname))
                n_saved += 1
    return n_saved


if __name__ == "__main__":
    print("=" * 60)
    print("Step 1 — Won et al. RSVP -> CSV")
    print("=" * 60)
    won_total = 0
    won_subs  = 0
    sub_dirs  = sorted(
        d for d in os.listdir(BIDS_ROOT)
        if d.startswith("sub-") and os.path.isdir(os.path.join(BIDS_ROOT, d))
    )
    for idx, sub_id in enumerate(sub_dirs, 1):
        sub_label = f"S{idx:02d}"
        n = process_won_subject(
            os.path.join(BIDS_ROOT, sub_id), sub_label, WON_OUT
        )
        if n > 0:
            print(f"  {sub_id} -> {sub_label}: {n} trials")
            won_total += n
            won_subs  += 1
    print(f"\nWon: {won_subs} subjects, {won_total} total trials saved")

    print()
    print("=" * 60)
    print("Step 2 — Zhang 57-ch -> 32-ch subset")
    print("=" * 60)
    zhang_subs = sorted(
        p for p in os.listdir(ZHANG_ROOT)
        if p.startswith("S") and os.path.isdir(os.path.join(ZHANG_ROOT, p))
    )
    for participant in zhang_subs:
        n = reduce_zhang_subject(participant)
        print(f"  {participant}: {n} trials re-saved")

    print("\nDone.")
    print(f"  Won output     -> {WON_OUT}")
    print(f"  Zhang (32-ch)  -> {ZHANG_REDUCED_OUT}")
