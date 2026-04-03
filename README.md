# Red-Team Evaluation of an EEG Biometric Authenticator

## Overview

This project applies a multi-armed bandit algorithm to automatically discover effective attacks against an EEG-based biometric authentication system. The idea is as follows: instead of manually testing attack methods one by one, we let a bandit algorithm figure out which ones work best.


## Background 

The target system is an EEG biometric authenticator: it identifies a person by the pattern of their brain signals. It works by projecting EEG trials onto a Riemannian manifold (via log-covariance matrices), building a per-subject template at enrolment, and then accepting/rejecting a verification attempt.

The dataset has 15 participants recorded over 4 sessions on different days (Day 1, 7, 80, 200). All evaluation is strictly within-session: enrolment and test data come from the same day for each subject to avoid any data leakage across days.

There are 8 attack strategies ("arms") that perturb EEG signals in different ways: Gaussian noise, correlated noise, temporal distortion, frequency manipulation, covariance interpolation, replay with jitter, channel coupling, and eigenvalue perturbation. 


## Where do I need feedback

I only need you to evaluate the UCB implementation. Everything else (aka Riemannian geometry, EEG signal processing, attack methods) is out of scope for your research field, so feel free to ignore it.


### The two relevant files are:

- `ucb.py`, the UCB1 bandit
- `attacks.py` — where the bandit is used

The `_attack_loop()` function is the main loop (1000 iterations per day). Each iteration follows the standard bandit pattern:

1. Call `ucb.select()` → get the arm to try
2. Sample a random victim EEG trial from the test pool
3. Apply the selected attack to perturb the signal
4. Submit the perturbed signal to the authenticator, claiming to be a different subject (this directly measures the False Acceptance Rate)
5. Call `ucb.update(arm, reward)` with the binary outcome

A fresh `UCBSelector` instance is created per day, each day is an independent bandit run.

The `run_redteam()` function is the outer loop that handles data loading, authenticator fitting, and calling `_attack_loop()` for each day.