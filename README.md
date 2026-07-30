# Red-Team Evaluation of an EEG Biometric Authenticator

EEG biometric authentication via Riemannian log-covariance features, red-teamed with a UCB multi-armed bandit, a black-box VAE, and a surrogate-gradient GAN across 8 datasets and 3 EEG paradigms.

## Setup

```bash
pip install numpy scipy pandas matplotlib torch mne
```

Data goes in `data/raw/<dataset>/` and `data/preprocessed/<dataset>/`. Run preprocessing scripts first, then experiments.

## Structure

```
preprocess/     preprocessing scripts per dataset
attacks/        vae.py (black-box REINFORCE), sgan.py (surrogate-gradient minimax)
experiments/    bandit_all.py (UCB on all 8 datasets), run_vae.py, run_sgan.py,
                resting.py, motor_imagery.py, lambda_sweep.py, feature_ablation.py,
                statistics.py
figures/        threshold_sweep.py, plot_arms.py
results/        Wilcoxon test outputs and arm FAR figures
```

## Datasets

(i) zhang (RSVP), won (P300-speller)
(ii) resting state, eo & ec: tanner, cogbci
(iii) motor imagery: cho, lee

## Authenticator

Nearest-centroid classifier on log-covariance vectors (Riemannian manifold). Enrollment builds per-subject templates; verification uses cosine similarity against the claimed identity's template. Threshold set at the p-th percentile of genuine scores (p=10 default, EER reported).

## Attack arms (UCB bandit)

8 perturbation strategies, selected by UCB1 with dual reward r = (1-$\lambda$)·accepted + $\lambda$ x (1-budget):

| Arm | Method |
|-----|--------|
| a_0 | Gaussian noise |
| a1 | Spatially correlated noise |
| a2 | Temporal distortion (shuffle/crop/shift) |
| a3 | Frequency band gain manipulation |
| a4 | Covariance interpolation with impostor trial |
| a5 | Replay with jitter |
| a6 | Channel coupling |
| a7 | Eigenvalue perturbation |

## Experiments

```bash
# UCB bandit on all 8 datasets
python experiments/bandit_all.py

# Black-box VAE attack
python experiments/run_vae.py

# Surrogate-gradient GAN
python experiments/run_sgan.py

# Lambda ablation
python experiments/lambda_sweep.py

# Feature space comparison (Riemannian vs Euclidean vs PSD)
python experiments/feature_ablation.py

# Wilcoxon significance tests
python experiments/statistics.py
```

All scripts write results to `results_<dataset>/` directories.
