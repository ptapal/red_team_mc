"""
Preprocess Cho et al. 2017 motor imagery EEG into the same CSV format used by load_sessions() in helpers.py.

Output directory:
    <RTMC_DATA>/preprocessed/cho/

Structure: {out_dir}/{sub_id}/session1/block/trial_{NNN:03d}.csv
"""

import os
import numpy as np
import pandas as pd
import scipy.io
import mne

mne.set_log_level("WARNING")

from paths import RAW_ROOT, PREP_ROOT
from config import SFREQ_TARGET, WON_CHANNELS_32, BANDPASS_WIDE

MAT_DIR  = str(RAW_ROOT / "cho/gigadb-datasets/live/pub/10.5524/100001_101000/100295/mat_data")
OUT_DIR  = str(PREP_ROOT / "cho")

SFREQ_IN  = 512
SFREQ_OUT = SFREQ_TARGET
L_FREQ, H_FREQ = BANDPASS_WIDE   # 1-30 Hz: motor imagery

N_EEG     = 64       # first 64 channels are EEG; 65-68 are EMG
PRE_SAMP  = 1024     # samples before cue onset at 512 Hz (= 2000 ms)
IMG_SAMPS = 2048     # 0 to +4 s post-cue at 512 Hz

# All 64 Cho EEG channel names (used to build the MNE RawArray)
CHO_ALL_CH = [
    "Fp1",  "AF7",  "AF3",  "F1",   "F3",   "F5",   "F7",   "FT7",
    "FC5",  "FC3",  "FC1",  "C1",   "C3",   "C5",   "T7",   "TP7",
    "CP5",  "CP3",  "CP1",  "P1",   "P3",   "P5",   "P7",   "P9",
    "PO7",  "PO3",  "O1",   "Iz",   "Oz",   "POz",  "Pz",   "CPz",
    "Fpz",  "Fp2",  "AF8",  "AF4",  "AFz",  "Fz",   "F2",   "F4",
    "F6",   "F8",   "FT8",  "FC6",  "FC4",  "FC2",  "FCz",  "Cz",
    "C2",   "C4",   "C6",   "T8",   "TP8",  "CP6",  "CP4",  "CP2",
    "P2",   "P4",   "P6",   "P8",   "P10",  "PO8",  "PO4",  "O2",
]

# 32-channel Won subset ; matches RSVP pair (C=32, D=528)
WON_CHANNELS = WON_CHANNELS_32

SUBJECTS = [f"s{i:02d}" for i in range(1, 53)]
MIN_TRIALS = 10



def load_and_process(mat_path):
    """
    Load a Cho .mat file, filter+resample the concatenated continuous data
    (one pass per condition, avoiding per-trial edge effects), then reshape
    into imagery windows.

    Returns pooled array of shape (n_ch, n_trials_total, samps_out) or None.
    """
    mat = scipy.io.loadmat(mat_path, simplify_cells=True)
    eeg = mat["eeg"]

    n_trials = int(eeg["n_imagery_trials"])
    frame    = np.array(eeg["frame"], dtype=np.int32)
    pre_s    = int(abs(int(frame[0])) * SFREQ_IN // 1000)   # 1024
    post_s   = int(int(frame[1])      * SFREQ_IN // 1000)   # 2560
    total_s  = pre_s + post_s                                # 3584

    if pre_s != PRE_SAMP:
        print(f"  WARNING: unexpected pre_samp={pre_s}")

    # Filter each condition's concatenated signal once (avoids per-trial edge effects)
    results = []
    for key in ("imagery_left", "imagery_right"):
        data = np.array(eeg[key], dtype=np.float64)[:N_EEG]  # (64, n_trials*total_s)

        # filter the full concatenated signal, then pick the 32 Won channels
        info = mne.create_info(ch_names=CHO_ALL_CH, sfreq=SFREQ_IN, ch_types="eeg")
        raw  = mne.io.RawArray(data, info, verbose=False)
        raw.pick_channels(WON_CHANNELS, ordered=True)
        raw.filter(L_FREQ, H_FREQ, method="iir",
                   iir_params={"order": 4, "ftype": "butter"}, verbose=False)
        raw.resample(SFREQ_OUT, verbose=False)
        data_filt = raw.get_data()  # (32, n_trials * total_s_resampled)

        n_out = len(WON_CHANNELS)
        # resampled trial length
        total_s_r = round(total_s * SFREQ_OUT / SFREQ_IN)   # 1750
        pre_s_r   = round(pre_s   * SFREQ_OUT / SFREQ_IN)   # 500
        img_s_r   = round(IMG_SAMPS * SFREQ_OUT / SFREQ_IN) # 1000

        epochs = data_filt.reshape(n_out, n_trials, total_s_r)         # (32, 100, 1750)
        windows = epochs[:, :, pre_s_r : pre_s_r + img_s_r]            # (32, 100, 1000)
        results.append(windows)

    pooled = np.concatenate(results, axis=1)   # (32, 200, 1000)
    return pooled



def main():
    total_trials = 0
    skipped      = 0

    for sub in SUBJECTS:
        mat_path = os.path.join(MAT_DIR, f"{sub}.mat")
        if not os.path.exists(mat_path):
            print(f"  SKIP (no file): {mat_path}")
            skipped += 1
            continue

        try:
            pooled = load_and_process(mat_path)  # (32, 200, 1000)
        except Exception as e:
            print(f"  ERROR {sub}: {e}")
            skipped += 1
            continue

        n_trials = pooled.shape[1]
        if n_trials < MIN_TRIALS:
            print(f"  SKIP {sub}: only {n_trials} trials")
            skipped += 1
            continue

        out_block = os.path.join(OUT_DIR, sub, "session1", "block")
        os.makedirs(out_block, exist_ok=True)

        for i in range(n_trials):
            trial = pooled[:, i, :]   # (32, 1000)
            df = pd.DataFrame(trial, index=WON_CHANNELS,
                              columns=np.arange(trial.shape[1]))
            df.to_csv(os.path.join(out_block, f"trial_{i+1:03d}.csv"))

        total_trials += n_trials
        print(f"  {sub}: {n_trials} trials saved")

    print(f"\nDone. Total trials: {total_trials}, subjects skipped: {skipped}")


if __name__ == "__main__":
    main()
