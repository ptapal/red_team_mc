# Red-Team Evaluation of an EEG Biometric Authenticator

EEG biometric authentication via Riemannian log-covariance features, red-teamed with a
UCB multi-armed bandit, a black-box VAE, and a surrogate-gradient GAN across 8 recording
streams and 3 EEG paradigms.

Code for *Adversarial Vulnerabilities of Neural Biomarker Identification Systems*
(Tapal & Bagley).

## Setup

```bash
git clone https://github.com/ptapal/red_team_mc.git
cd red_team_mc
pip install -e .
```

The editable install puts the project's top-level modules on the import path, so the
scripts run from any working directory.

## Data

Datasets are not redistributed here. Download each one from its source, then arrange
them as below.

| Key | Paradigm | Source |
|-----|----------|--------|
| `zhang` | RSVP | [10.1038/s41597-025-05378-x](https://doi.org/10.1038/s41597-025-05378-x) |
| `won` | RSVP / P300 speller | [10.1038/s41597-022-01509-w](https://doi.org/10.1038/s41597-022-01509-w) |
| `wang` (EO/EC) | resting state | [10.1038/s41597-022-01607-9](https://doi.org/10.1038/s41597-022-01607-9), OpenNeuro `ds004148` |
| `cogbci` (EO/EC) | resting state | [10.1038/s41597-022-01898-y](https://doi.org/10.1038/s41597-022-01898-y) |
| `cho` | motor imagery | [10.1093/gigascience/gix034](https://doi.org/10.1093/gigascience/gix034), GigaDB `10.5524/100295` |
| `lee` | motor imagery | [10.1093/gigascience/giz002](https://doi.org/10.1093/gigascience/giz002), GigaDB `10.5524/100542` |

Expected layout:

```
data/
  raw/           downloaded datasets, one directory per source
  preprocessed/  written by the preprocess/ scripts
```

`data/` next to this README is the default. To keep the datasets elsewhere, set:

```bash
export RTMC_DATA=/path/to/your/data
```

All dataset paths derive from that one root; see `paths.py`.

## Running

Preprocess first, once per dataset:

```bash
python preprocess/zhang_won.py     # RSVP: Zhang and Won
python preprocess/wang.py          # resting state, eyes open and closed
python preprocess/cogbci.py        # resting state, eyes open and closed
python preprocess/cho.py           # motor imagery
python preprocess/lee.py           # motor imagery
```

Then the experiments:

```bash
python experiments/bandit_all.py       # UCB bandit, nearest-centroid, all 8 streams
python experiments/run_gnb.py          # Gaussian naive Bayes
python experiments/run_svm.py          # linear SVM
python experiments/run_vae.py          # black-box VAE attacker
python experiments/run_sgan.py         # surrogate-gradient GAN attacker
python experiments/cross_condition.py  # cross-paradigm transfer
python experiments/cross_session.py    # across enrollment sessions
python experiments/lambda_sweep.py     # reward trade-off ablation
python experiments/feature_ablation.py # Riemannian vs Euclidean vs PSD
python experiments/attacker_lift.py    # attacker lift table
python experiments/statistics.py       # Wilcoxon tests
```

Results are written under `results/`. Note that `attacker_lift.py`,
`feature_ablation.py`, `motor_imagery.py` and `resting.py` execute on import, so import
them only if you intend to run them.

## Layout

```
paths.py        dataset roots, overridable with RTMC_DATA
config.py       seeds, arm names, and preprocessing constants shared across datasets
datasets.py     the 8 recording streams: labels, participants, sessions
helpers.py      session loading
utils.py        covariance features and cosine similarity
arms.py         the 8 perturbation arms
ucb.py          UCB1
preprocess/     one script per dataset
attacks/        vae.py (black-box REINFORCE), sgan.py (surrogate-gradient minimax)
experiments/    attack runs, ablations, statistics
figures/        threshold sweeps and arm comparison plots
results/        committed outputs: CSVs and figures
```

## Environments used for the published results

Preprocessing and the experiments were run in two environments:

| Stage | Python | Key versions |
|-------|--------|--------------|
| `preprocess/` | 3.10.18 | mne 1.10.0 |
| `experiments/`, `figures/` | 3.13.6 | numpy 2.5.1, scipy 1.18.0, pandas 3.0.5, matplotlib 3.11.1, scikit-learn 1.9.0, torch 2.13.0 |

`pyproject.toml` pins all of these together. The split is historical, not a
requirement.

## Authenticator

Log-covariance vectors on the SPD manifold (spatial covariance, Log-Euclidean map,
upper triangle, l2-normalised). Enrollment builds per-subject templates; verification
scores a trial against the claimed identity by cosine similarity. The decision threshold
is the p-th percentile of genuine scores (p=10 by default; EER also reported).
Nearest-centroid, Gaussian naive Bayes, and linear SVM are each evaluated.

## Attack arms

Eight perturbation strategies, selected by UCB1 with the dual reward
r = (1-lambda)*accepted + lambda*(1-budget):

| Arm | Method |
|-----|--------|
| a0 | Gaussian noise |
| a1 | Spatially correlated noise |
| a2 | Temporal distortion (shuffle/crop/shift) |
| a3 | Frequency band gain manipulation |
| a4 | Covariance interpolation with impostor trial |
| a5 | Replay with jitter |
| a6 | Channel coupling |
| a7 | Eigenvalue perturbation |

## Preprocessing note

Bandpass edges differ by paradigm and are not interchangeable: RSVP is filtered
1-10 Hz, resting state and motor imagery 1-30 Hz. Both are defined in `config.py` as
`BANDPASS_RSVP` and `BANDPASS_WIDE`.

## License

MIT, see `LICENSE`. The datasets carry their own licences; check each source before
redistributing anything derived from them.
