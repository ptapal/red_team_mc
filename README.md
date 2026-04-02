# Red-Team Evaluation of an EEG Biometric Authenticator

## Overview

This project applies a **multi-armed bandit (UCB1)** algorithm to automatically discover effective attacks against an EEG-based biometric authentication system. The idea is simple: instead of manually testing attack methods one by one, we let a bandit algorithm figure out which ones work best.

---

## Background (for context — not what you need to grade)

The target system is an **EEG biometric authenticator**: it identifies a person by the pattern of their brain signals. It works by projecting EEG trials onto a Riemannian manifold (via log-covariance matrices), building a per-subject template at enrolment, and then accepting/rejecting a verification attempt via cosine similarity.

The dataset has 15 participants recorded over 4 sessions on different days (Day 1, 7, 80, 200). All evaluation is strictly **within-session** — enrolment and test data come from the same day for each subject — to avoid any data leakage across days.

There are 8 attack strategies ("arms") that perturb EEG signals in different ways: Gaussian noise, correlated noise, temporal distortion, frequency manipulation, covariance interpolation, replay with jitter, channel coupling, and eigenvalue perturbation. The details of these don't matter for what you need to check.

---

## What You Need to Look At

**You only need to evaluate the UCB implementation.** Everything else (Riemannian geometry, EEG signal processing, attack methods) is out of scope for the IA and Games course — feel free to ignore it.

### The two relevant files are:

---

### `ucb.py` — the UCB1 bandit

This is the core. The `UCBSelector` class implements the standard UCB1 algorithm:

- **Initialisation**: `counts` and `rewards` arrays, one entry per arm (8 arms total).
- **Warm-up**: each arm is pulled once before UCB scores are used, to avoid division by zero and ensure at least one observation per arm.
- **Selection** (`select()`): after the warm-up, the arm with the highest UCB score is chosen:

$$\text{UCB}(a) = \hat{\mu}_a + c \cdot \sqrt{\frac{\ln t}{N_a}}$$

where $\hat{\mu}_a$ is the empirical mean reward of arm $a$, $t$ is the total number of pulls so far, $N_a$ is how many times arm $a$ has been pulled, and $c = \sqrt{2}$ is the exploration constant.

- **Update** (`update(arm, reward)`): increments the count and accumulates the reward for the chosen arm. Reward is binary: 1 if the attack succeeded (authenticator was fooled), 0 if not.

---

### `attacks.py` — where the bandit is used

The `_attack_loop()` function is the main loop (1000 iterations per day). Each iteration follows the standard bandit pattern:

1. Call `ucb.select()` → get the arm to try
2. Sample a random victim EEG trial from the test pool
3. Apply the selected attack to perturb the signal
4. Submit the perturbed signal to the authenticator, claiming to be a *different* subject (this directly measures the False Acceptance Rate)
5. Call `ucb.update(arm, reward)` with the binary outcome

A fresh `UCBSelector` instance is created per day — each day is an independent bandit run.

The `run_redteam()` function is the outer loop that handles data loading, authenticator fitting, and calling `_attack_loop()` for each day.

---

## File Map

| File | What it is | Do you need it? |
|------|-----------|-----------------|
| `ucb.py` | UCB1 bandit | **Yes — main thing to check** |
| `attacks.py` | Bandit loop + orchestration | **Yes — shows how UCB is used** |
| `arms.py` | 8 attack methods | No |
| `auth.py` | Riemannian authenticator | No |
| `utils.py` | Signal processing utilities | No |
| `helpers.py` | Data loading / session splits | No |
| `plots.py` | Result visualisation | No |
| `config.py` | Constants (n_iter=1000, c=√2, etc.) | Optional — quick reference |

---

## Key Design Choices

- **`c = √2`**: standard UCB1 exploration constant, set in `config.py` and passed into `UCBSelector`.
- **Warm-up before UCB**: each arm gets exactly one forced pull before the UCB formula kicks in. This is handled in `select()` by checking `if t < n_arms`.
- **One bandit per day**: the bandit state is not shared across days. This is intentional — each day is treated as an independent evaluation.
- **Binary reward**: reward is 1 (success) or 0 (failure), so the empirical mean converges to the False Acceptance Rate (FAR) for each arm.
