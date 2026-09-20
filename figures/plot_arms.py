"""
Generate paper-ready per-arm FAR bar charts at p=10 for all six datasets.

Fixed arm order (a0..a7) with names matching Table 1 in the paper.
Abandoned arms (FAR=0) shown in light grey to distinguish from active arms.
Saves arm_far_p10.png in each results directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datasets import get_datasets

ARM_PAPER_NAMES = [
    "Gaussian\nnoise",
    "Correlated\nnoise",
    "Temporal\ndistortion",
    "Frequency\nmanip.",
    "Covariance\ninterp.",
    "Replay\n+ jitter",
    "Channel\ncoupling",
    "Eigenvalue\nperturb.",
]

# CSV column name -> arm index mapping (matches config.py ARM_NAMES order)
CSV_ARM_NAMES = [
    "Gaussian Noise",
    "Correlated Noise",
    "Temporal Distortion",
    "Freq Manipulation",
    "Cov Interpolation",
    "Replay + Jitter",
    "Channel Coupling",
    "Eigenvalue Perturb",
]

DATASETS = [(v["out_dir"], v["label"]) for v in get_datasets("results").values()]

P_REF = 10

ACTIVE_COLOR   = "#2E86AB"   # steel blue
INACTIVE_COLOR = "#C8C8C8"   # light grey (abandoned by bandit)
ERR_COLOR      = "#1a5276"   # darker blue for error caps

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "figure.dpi":       300,
})

def plot_dataset(results_dir, title):
    csv_path = os.path.join(results_dir, "adv_far_std.csv")
    if not os.path.exists(csv_path):
        print(f"  SKIP (no CSV): {csv_path}")
        return

    df = pd.read_csv(csv_path)
    row = df[df["p"] == P_REF]
    if row.empty:
        print(f"  SKIP (p={P_REF} not found): {csv_path}")
        return
    row = row.iloc[0]

    means = [row[f"arm_{n}_mean"] for n in CSV_ARM_NAMES]
    stds  = [row[f"arm_{n}_std"]  for n in CSV_ARM_NAMES]

    x      = np.arange(len(ARM_PAPER_NAMES))
    colors = [INACTIVE_COLOR if m < 1e-6 else ACTIVE_COLOR for m in means]
    ecols  = [INACTIVE_COLOR if m < 1e-6 else ERR_COLOR    for m in means]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    bars = ax.bar(x, means, color=colors, width=0.6, zorder=3)
    for xi, (m, s, ec) in enumerate(zip(means, stds, ecols)):
        if m > 1e-6:
            ax.errorbar(xi, m, yerr=s, fmt="none", ecolor=ec,
                        elinewidth=1.2, capsize=3, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(ARM_PAPER_NAMES, fontsize=7.5, linespacing=1.2)
    ax.set_ylabel("Adversarial FAR", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.5,
            alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # label value above each non-zero bar
    for bar, m, s in zip(bars, means, stds):
        if m > 1e-6:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                m + s + 0.02,
                f"{m:.2f}",
                ha="center", va="bottom", fontsize=6.5, color="#222222",
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.4)
    out_path = os.path.join(results_dir, "arm_far_p10.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    for rel_dir, title in DATASETS:
        full_dir = os.path.join(base, rel_dir)
        print(f"{title}")
        plot_dataset(full_dir, title)
    print("Done.")
