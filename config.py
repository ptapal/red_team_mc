import math

s_freq = 250

DEFAULT_N_ITER     = 1000
DEFAULT_C_UCB      = math.sqrt(2)
DEFAULT_OUTPUT_DIR = "results"
SEED               = 42
ENROL_FRAC         = 0.7
MIN_TRIALS_SESSION = 4
N_ARMS             = 8

# ---------------------------------------------------------------------------
# Preprocessing constants shared by every dataset
# ---------------------------------------------------------------------------

SFREQ_TARGET = 250      # Hz, common sampling rate after resampling
TRIAL_SAMPLES = 162     # samples per trial/window at SFREQ_TARGET (= 648 ms)
MIN_WINDOWS = 5         # discard resting-state recordings with fewer windows

# Won montage, 32 channels. Fixed order: pick_channels(ordered=True) and the
# per-trial CSV row index both depend on it.
WON_CHANNELS_32 = [
    "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7",  "C3",
    "CP1", "CP5", "P7", "P3", "Pz",  "PO3", "O1",  "Oz",
    "O2",  "PO4", "P4", "P8", "CP6", "CP2", "C4",  "T8",
    "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz",  "Cz",
]

# CP5 is not recorded for COG-BCI sub-17/25/26/28/29, so it is dropped for both
# resting-state datasets to keep their feature spaces identical.
WON_CHANNELS_31 = [c for c in WON_CHANNELS_32 if c != "CP5"]

# Bandpass edges differ by paradigm and are NOT interchangeable.
# RSVP is 1-10 Hz; resting state and motor imagery are 1-30 Hz.
BANDPASS_RSVP = (1.0, 10.0)
BANDPASS_WIDE = (1.0, 30.0)

ARM_NAMES = [
    "Gaussian Noise",
    "Correlated Noise",
    "Temporal Distortion",
    "Freq Manipulation",
    "Cov Interpolation",
    "Replay + Jitter",
    "Channel Coupling",
    "Eigenvalue Perturb",
]
