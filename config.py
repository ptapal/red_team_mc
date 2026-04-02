import math
import torch
import torch.nn as nn

s_freq = 250

DEFAULT_N_ITER     = 1000
DEFAULT_C_UCB      = math.sqrt(2)
DEFAULT_OUTPUT_DIR = "results" # or "results_hybrid"
SEED               = 42
ENROL_FRAC         = 0.7   
MIN_TRIALS_SESSION = 4     
N_ARMS             = 8

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

# hyperparameters for hybrid
LATENT_DIM    = 128
EPOCHS        = 200
WARMUP_EPOCHS = 80
PATIENCE      = 20
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"