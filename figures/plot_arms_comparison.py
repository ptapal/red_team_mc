import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

ARM_PAPER_NAMES = [
    "Gaussian\nnoise", "Correlated\nnoise", "Temporal\ndistortion",
    "Frequency\nmanip.", "Covariance\ninterp.", "Replay\n+ jitter",
    "Channel\ncoupling", "Eigenvalue\nperturb.",
]
CSV_ARM_NAMES = [
    "Gaussian Noise", "Correlated Noise", "Temporal Distortion",
    "Freq Manipulation", "Cov Interpolation", "Replay + Jitter",
    "Channel Coupling", "Eigenvalue Perturb",
]

P_REF = 10
ACTIVE_COLOR   = "#2E86AB"
INACTIVE_COLOR = "#C8C8C8"
ERR_COLOR      = "#1a5276"
GNB_ACTIVE_COLOR = "#2E8B57"
GNB_ERR_COLOR    = "#1D5C38"
SVM_ACTIVE_COLOR = "#C0392B"
SVM_ERR_COLOR    = "#7B241C"

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "figure.dpi":       300,
})

DATASETS = {
    "zhang":     ("results/zhang/adv_far_std.csv",      "results/gnb/zhang/adv_far_std.csv",     "results/svm/zhang/adv_far_std.csv",     "Zhang (RSVP)",             "results/zhang"),
    "won":       ("results/won/adv_far_std.csv",        "results/gnb/won/adv_far_std.csv",       "results/svm/won/adv_far_std.csv",       "Won (RSVP)",               "results/won"),
    "tanner_eo": ("results/wang/adv_far_std_eo.csv",     "results/gnb/tanner_eo/adv_far_std.csv", "results/svm/tanner_eo/adv_far_std.csv", "Wang-EO (resting)",        "results/wang"),
    "tanner_ec": ("results/wang/adv_far_std_ec.csv",     "results/gnb/tanner_ec/adv_far_std.csv", "results/svm/tanner_ec/adv_far_std.csv", "Wang-EC (resting)",        "results/wang"),
    "cogbci_eo": ("results/COG-BCI/adv_far_std_eo.csv",  "results/gnb/cogbci_eo/adv_far_std.csv", "results/svm/cogbci_eo/adv_far_std.csv", "COG-BCI-EO (resting)",     "results/COG-BCI"),
    "cogbci_ec": ("results/COG-BCI/adv_far_std_ec.csv",  "results/gnb/cogbci_ec/adv_far_std.csv", "results/svm/cogbci_ec/adv_far_std.csv", "COG-BCI-EC (resting)",     "results/COG-BCI"),
    "cho":       ("results/cho/adv_far_std.csv",         "results/gnb/cho/adv_far_std.csv",       "results/svm/cho/adv_far_std.csv",       "Cho (motor imagery)",      "results/cho"),
    "lee":       ("results/lee/adv_far_std.csv",         "results/gnb/lee/adv_far_std.csv",       "results/svm/lee/adv_far_std.csv",       "Lee (motor imagery)",      "results/lee"),
}


def _load_p10(csv_path):
    df = pd.read_csv(csv_path)
    row = df[df["p"] == P_REF]
    if row.empty:
        return None, None
    row = row.iloc[0]
    means = [row[f"arm_{n}_mean"] for n in CSV_ARM_NAMES]
    stds  = [row[f"arm_{n}_std"]  for n in CSV_ARM_NAMES]
    return means, stds


def plot_comparison(ds_key, nc_csv, gnb_csv, svm_csv, label, out_dir):
    if not os.path.exists(nc_csv) or not os.path.exists(gnb_csv) or not os.path.exists(svm_csv):
        print(f"  SKIP {ds_key}: missing CSV ({nc_csv}, {gnb_csv}, or {svm_csv})")
        return

    nc_means,  nc_stds  = _load_p10(nc_csv)
    gnb_means, gnb_stds = _load_p10(gnb_csv)
    svm_means, svm_stds = _load_p10(svm_csv)
    if nc_means is None or gnb_means is None or svm_means is None:
        print(f"  SKIP {ds_key}: p={P_REF} not found")
        return

    x  = np.arange(len(ARM_PAPER_NAMES))
    w  = 0.25

    nc_colors  = [INACTIVE_COLOR if m < 1e-6 else ACTIVE_COLOR     for m in nc_means]
    nc_ecols   = [INACTIVE_COLOR if m < 1e-6 else ERR_COLOR        for m in nc_means]
    gnb_colors = [INACTIVE_COLOR if m < 1e-6 else GNB_ACTIVE_COLOR for m in gnb_means]
    gnb_ecols  = [INACTIVE_COLOR if m < 1e-6 else GNB_ERR_COLOR    for m in gnb_means]
    svm_colors = [INACTIVE_COLOR if m < 1e-6 else SVM_ACTIVE_COLOR for m in svm_means]
    svm_ecols  = [INACTIVE_COLOR if m < 1e-6 else SVM_ERR_COLOR    for m in svm_means]

    fig, ax = plt.subplots(figsize=(9.5, 3.2))

    bars_nc  = ax.bar(x - w, nc_means,  w, color=nc_colors,  zorder=3)
    bars_gnb = ax.bar(x,     gnb_means, w, color=gnb_colors, zorder=3)
    bars_svm = ax.bar(x + w, svm_means, w, color=svm_colors, zorder=3)

    for xi, (m, s, ec) in enumerate(zip(nc_means, nc_stds, nc_ecols)):
        if m > 1e-6:
            ax.errorbar(xi - w, m, yerr=s, fmt="none", ecolor=ec,
                        elinewidth=1.2, capsize=3, zorder=4)
    for xi, (m, s, ec) in enumerate(zip(gnb_means, gnb_stds, gnb_ecols)):
        if m > 1e-6:
            ax.errorbar(xi, m, yerr=s, fmt="none", ecolor=ec,
                        elinewidth=1.2, capsize=3, zorder=4)
    for xi, (m, s, ec) in enumerate(zip(svm_means, svm_stds, svm_ecols)):
        if m > 1e-6:
            ax.errorbar(xi + w, m, yerr=s, fmt="none", ecolor=ec,
                        elinewidth=1.2, capsize=3, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(ARM_PAPER_NAMES, fontsize=7.5, linespacing=1.2)
    ax.set_ylabel("Adversarial FAR", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f"{label} — per-arm adversarial FAR at $p=10$", fontsize=9.5, pad=28)

    for bar, m, s in zip(bars_nc, nc_means, nc_stds):
        if m > 1e-6:
            ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.02,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=6, color="#222222")
    for bar, m, s in zip(bars_gnb, gnb_means, gnb_stds):
        if m > 1e-6:
            ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.02,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=6, color="#222222")
    for bar, m, s in zip(bars_svm, svm_means, svm_stds):
        if m > 1e-6:
            ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.02,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=6, color="#222222")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_handles = [
        Patch(facecolor=ACTIVE_COLOR,     label="Nearest-centroid"),
        Patch(facecolor=GNB_ACTIVE_COLOR, label="GNB"),
        Patch(facecolor=SVM_ACTIVE_COLOR, label="SVM"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, fontsize=8, frameon=False, borderaxespad=0.3)

    plt.tight_layout(pad=0.4)

    out_path = os.path.join(out_dir, "arm_comparison_p10.png")
    suffix = {"tanner_eo": "_eo", "tanner_ec": "_ec",
              "cogbci_eo": "_eo", "cogbci_ec": "_ec"}.get(ds_key, "")
    if suffix:
        out_path = os.path.join(out_dir, f"arm_comparison_p10{suffix}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    for ds_key, (nc_csv, gnb_csv, svm_csv, label, out_dir) in DATASETS.items():
        print(label)
        plot_comparison(ds_key, nc_csv, gnb_csv, svm_csv, label, out_dir)
    print("Done.")
