import matplotlib.pyplot as plt
import os
from config import ARM_NAMES, N_ARMS
import numpy as np

def _plot_day(ucb, far_per_arm, far_vs_strength, logs, day, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"EEG Red-Teaming — {day} (within-session)",
                 fontsize=14, fontweight="bold")

    # (A) FAR per arm
    ax     = axes[0, 0]
    names  = list(far_per_arm.keys())
    fars   = list(far_per_arm.values())
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(names)))
    bars   = ax.barh(names, fars, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(x=np.mean(fars), color="gray", linestyle="--",
               linewidth=1, label="mean FAR")
    for bar, v in zip(bars, fars):
        ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, max(fars) * 1.25 + 0.05)
    ax.set_xlabel("False Acceptance Rate")
    ax.set_title("(A) FAR per Attack Arm")
    ax.legend(fontsize=8)

    # (B) UCB pull counts
    ax = axes[0, 1]
    ax.bar(ARM_NAMES, ucb.counts, color="#4C72B0", edgecolor="white")
    ax.set_xticklabels(ARM_NAMES, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Pull count")
    ax.set_title("(B) UCB Arm Selection Frequency")

    # (C) Cumulative FAR
    ax       = axes[1, 0]
    accepted = np.array([l["accepted"] for l in logs], dtype=float)
    cum_far  = np.cumsum(accepted) / (np.arange(len(accepted)) + 1)
    ax.plot(cum_far, color="#2196F3", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative FAR")
    ax.set_title("(C) Cumulative FAR Over Iterations")
    ax.grid(True, alpha=0.3)

    # (D) FAR vs perturbation strength (top-3 arms)
    ax   = axes[1, 1]
    top3 = sorted(far_per_arm, key=far_per_arm.get, reverse=True)[:3]
    for color, name in zip(["#E74C3C", "#F39C12", "#2ECC71"], top3):
        arm_i = ARM_NAMES.index(name)
        d     = far_vs_strength.get(arm_i, {})
        if d:
            ax.plot(d["bin_mid"], d["far"], marker="o", label=name,
                    color=color, linewidth=2)
    ax.set_xlabel("Perturbation Strength (||delta|| / ||x||)")
    ax.set_ylabel("FAR")
    ax.set_title("(D) FAR vs Perturbation Strength (top-3 arms)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "redteam_results.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {path}")


def _plot_cross_day(all_results: dict, output_dir: str):
    """
    Aggregate plot: overall FAR and per-arm FAR heatmap across days.
    This is purely a post-hoc visualisation — it does not combine
    data across days, only their independently-computed FAR scores.
    """
    if len(all_results) < 2:
        return

    days        = sorted(all_results.keys())
    overall_far = [all_results[d]["overall_FAR"] for d in days]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cross-Day Attack Summary (within-session evaluation)",
                 fontsize=13, fontweight="bold")

    # Overall FAR by day
    ax = axes[0]
    ax.bar(days, overall_far, color="#2196F3", edgecolor="white")
    ax.set_ylabel("Overall FAR")
    ax.set_title("Overall FAR per Day")
    ax.set_ylim(0, max(overall_far) * 1.3 + 0.05)
    for i, v in enumerate(overall_far):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    # Per-arm FAR heatmap (rows=days, cols=arms)
    ax     = axes[1]
    matrix = np.array([[all_results[d]["far_per_arm"].get(n, 0.0)
                         for n in ARM_NAMES] for d in days])
    im     = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r",
                       vmin=0, vmax=max(matrix.max(), 1e-6))
    ax.set_xticks(range(N_ARMS))
    ax.set_xticklabels(ARM_NAMES, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(days)))
    ax.set_yticklabels(days)
    ax.set_title("Per-Arm FAR Heatmap (rows=days, cols=arms)")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="FAR")

    plt.tight_layout()
    path = os.path.join(output_dir, "cross_day_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nCross-day summary saved -> {path}")