import numpy as np
from scipy.signal import butter, sosfilt

class AttackArms:

    def __init__(self, rng: np.random.Generator, s_freq: int = 256):
        self.rng    = rng
        self.s_freq = s_freq

    def gaussian_noise(self, trial, sigma=None):
        if sigma is None:
            sigma = self.rng.uniform(0.005, 0.2)
        return trial + self.rng.normal(0, sigma, size=trial.shape), \
               {"arm": 0, "name": "gaussian_noise", "sigma": sigma}

    def correlated_noise(self, trial, sigma=None, n_components=None):
        n_ch, n_t = trial.shape
        if sigma is None:
            sigma = self.rng.uniform(0.01, 0.15)
        if n_components is None:
            n_components = int(self.rng.integers(1, max(2, n_ch // 3)))
        directions = self.rng.standard_normal((n_ch, n_components))
        directions /= np.linalg.norm(directions, axis=0, keepdims=True) + 1e-9
        weights    = self.rng.standard_normal((n_components, n_t))
        return trial + sigma * (directions @ weights), \
               {"arm": 1, "name": "correlated_noise",
                "sigma": sigma, "n_components": n_components}

    def temporal_distortion(self, trial, mode=None):
        modes = ["shuffle", "crop_pad", "shift"]
        if mode is None:
            mode = self.rng.choice(modes)
        n_ch, n_t = trial.shape
        out    = trial.copy()
        params = {"arm": 2, "name": "temporal_distortion", "mode": mode}

        if mode == "shuffle":
            seg_len = int(self.rng.integers(n_t // 10, n_t // 4))
            n_segs  = n_t // seg_len
            segs    = [trial[:, i*seg_len:(i+1)*seg_len] for i in range(n_segs)]
            self.rng.shuffle(segs)
            out = np.concatenate(segs, axis=1)
            if out.shape[1] < n_t:
                out = np.pad(out, ((0, 0), (0, n_t - out.shape[1])), mode="edge")
            params["seg_len"] = seg_len

        elif mode == "crop_pad":
            frac    = self.rng.uniform(0.7, 0.95)
            keep    = int(n_t * frac)
            start   = int(self.rng.integers(0, n_t - keep))
            cropped = trial[:, start:start + keep]
            out     = np.pad(cropped, ((0, 0), (0, n_t - keep)), mode="edge")
            params["frac"] = frac

        elif mode == "shift":
            shift = int(self.rng.integers(1, n_t // 5))
            out   = np.roll(trial, shift, axis=1)
            params["shift"] = shift

        return out[:, :n_t], params

    def frequency_manipulation(self, trial, gain=None):
        bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
                 "beta": (13, 30), "gamma": (30, 80)}
        band_name = self.rng.choice(list(bands.keys()))
        lo, hi    = bands[band_name]
        nyq       = self.s_freq / 2.0
        hi        = min(hi, nyq - 1)
        if gain is None:
            gain = self.rng.uniform(0.0, 3.0)
        try:
            sos      = butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
            band_sig = sosfilt(sos, trial, axis=1)
            out      = (trial - band_sig) + gain * band_sig
        except Exception:
            out = trial.copy()
        return out, {"arm": 3, "name": "freq_manipulation",
                     "band": band_name, "gain": gain}

    def covariance_interpolation(self, trial, impostor_trial, lam=None):
        if lam is None:
            lam = self.rng.uniform(0.0, 0.5)
        out = (1 - lam) * trial + lam * impostor_trial
        return out, {"arm": 4, "name": "cov_interpolation", "lambda": lam}

    def replay_jitter(self, trial, noise_std=None, shift_max=None):
        if noise_std is None:
            noise_std = self.rng.uniform(1e-4, 0.05)
        if shift_max is None:
            shift_max = int(self.rng.integers(0, 20))
        out   = trial + self.rng.normal(0, noise_std, size=trial.shape)
        shift = int(self.rng.integers(-shift_max, shift_max + 1)) if shift_max > 0 else 0
        out   = np.roll(out, shift, axis=1)
        return out, {"arm": 5, "name": "replay_jitter",
                     "noise_std": noise_std, "shift": shift}

    def channel_coupling(self, trial, alpha=None):
        n_ch = trial.shape[0]
        if alpha is None:
            alpha = self.rng.uniform(0.1, 1.0)
        i, j = self.rng.choice(n_ch, size=2, replace=False)
        out  = trial.copy()
        out[i] = trial[i] + alpha * trial[j]
        return out, {"arm": 6, "name": "channel_coupling",
                     "alpha": alpha, "ch_i": int(i), "ch_j": int(j)}

    def eigenvalue_perturbation(self, trial, mode=None, factor=None):
        modes = ["amplify", "flatten"]
        if mode is None:
            mode = self.rng.choice(modes)
        if factor is None:
            factor = self.rng.uniform(1.1, 3.0) if mode == "amplify" \
                     else self.rng.uniform(0.1, 0.9)

        cov        = np.cov(trial)
        vals, vecs = np.linalg.eigh(cov)
        vals       = np.maximum(vals, 1e-8)

        if mode == "amplify":
            k = max(1, int(self.rng.integers(1, len(vals) // 2 + 1)))
            vals[-k:] *= factor
        else:
            mean_val = vals.mean()
            vals     = vals + factor * (mean_val - vals)

        vals          = np.maximum(vals, 1e-8)
        cov_perturbed = vecs @ np.diag(vals) @ vecs.T

        try:
            out = self.rng.multivariate_normal(
                mean=np.zeros(trial.shape[0]),
                cov=cov_perturbed,
                size=trial.shape[1]
            ).T
            orig_std = trial.std(axis=1, keepdims=True) + 1e-9
            new_std  = out.std(axis=1, keepdims=True) + 1e-9
            out      = out * (orig_std / new_std)
        except np.linalg.LinAlgError:
            out = trial.copy()

        return out, {"arm": 7, "name": "eigenvalue_perturbation",
                     "mode": mode, "factor": factor}

    def apply(self, arm_idx, trial, impostor_trial=None):
        if arm_idx == 0:  return self.gaussian_noise(trial)
        if arm_idx == 1:  return self.correlated_noise(trial)
        if arm_idx == 2:  return self.temporal_distortion(trial)
        if arm_idx == 3:  return self.frequency_manipulation(trial)
        if arm_idx == 4:
            if impostor_trial is None:
                return self.gaussian_noise(trial)   # fallback
            return self.covariance_interpolation(trial, impostor_trial)
        if arm_idx == 5:  return self.replay_jitter(trial)
        if arm_idx == 6:  return self.channel_coupling(trial)
        if arm_idx == 7:  return self.eigenvalue_perturbation(trial)
        raise ValueError(f"Unknown arm index: {arm_idx}")
