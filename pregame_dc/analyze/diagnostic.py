"""Pregame PCA diagnostic.

For each game, take the 24-dim ask-price vector at t=-10 minutes (i.e.
seconds_since_game_start = -600). Compute the empirical covariance across
games and visualise:

    1. covariance (or correlation) matrix as a 24x24 colour plot
    2. eigenvalue spectrum (scree)
    3. top-K eigenvector loadings

`--standardize` flag runs PCA on the correlation matrix (every coordinate
divided by its std before SVD) instead of the raw covariance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from pregame_dc.constants import MARKET_LABELS

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FILES = {
    "self_collected": REPO_ROOT / "data/mlp_v3/self_collected_dataset.parquet",
    "telonex": REPO_ROOT / "data/mlp_v3/telonex_dataset.parquet",
}
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
T_TARGET = -600.0  # seconds_since_game_start = -10 minutes
SPLIT = "train"
LOG_FLOOR = 0.01  # 1¢, matches the CLOB minimum tick; clip before log()

X_COLS = [f"x_{i}" for i in range(24)]


def load_pregame_vectors(sources: list[str]) -> np.ndarray:
    rows = []
    for src in sources:
        f = DATA_FILES[src]
        df = pq.read_table(
            f, columns=["game_slug", "source", "split", "seconds_since_game_start", *X_COLS]
        ).to_pandas()
        df = df[(df["split"] == SPLIT) & (df["seconds_since_game_start"] == T_TARGET)]
        X = df[X_COLS].to_numpy(dtype=np.float64)
        # Drop fully stale books (every ask == 1.0).
        keep = ~(X == 1.0).all(axis=1)
        X = X[keep]
        rows.append(X)
        print(f"{f.name}: {X.shape[0]} games kept (split={SPLIT}, t={T_TARGET}s)")
    return np.concatenate(rows, axis=0)


def token_labels() -> list[str]:
    # Dataset layout (set in scripts/build_mlp_v2_dataset.py:pivot_per_game_data):
    #   x_0..x_11  = YES asks for canonical slots 0..11
    #   x_12..x_23 = NO  asks for canonical slots 0..11
    # Canonical slot order from strategies/convex_arb/analyze.py:MARKET_LABELS.
    return [f"{l} Y" for l in MARKET_LABELS] + [f"{l} N" for l in MARKET_LABELS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--standardize", action="store_true",
        help="Divide each coordinate by its std before PCA (PCA on the "
             "correlation matrix). Saves plots with a _std suffix.",
    )
    ap.add_argument(
        "--sources", nargs="+", default=["self_collected", "telonex"],
        choices=list(DATA_FILES.keys()),
        help="Which dataset(s) to include.",
    )
    ap.add_argument(
        "--log-prices", action="store_true",
        help="Use log(ask) (clipped at 1¢) instead of raw ask. Adds _log "
             "to the file suffix.",
    )
    args = ap.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    X = load_pregame_vectors(args.sources)
    n, d = X.shape
    print(f"\nAggregate: n={n} games, d={d}, sources={args.sources}")

    suffix_parts = []
    src_label = "+".join(args.sources)
    if args.sources != ["self_collected", "telonex"]:
        suffix_parts.append(src_label)

    if args.log_prices:
        X = np.log(np.clip(X, LOG_FLOOR, 1.0))
        suffix_parts.append("log")
        feature_label = "log(ask)"
    else:
        feature_label = "ask"

    print(f"per-feature mean range: [{X.mean(0).min():.3f}, {X.mean(0).max():.3f}]")
    print(f"per-feature std  range: [{X.std(0).min():.3f}, {X.std(0).max():.3f}]")

    if args.standardize:
        sd = X.std(axis=0, ddof=1)
        sd_safe = np.where(sd > 0, sd, 1.0)
        Xw = (X - X.mean(axis=0)) / sd_safe
        suffix_parts.append("std")
        mode_label = f"standardized {feature_label} (correlation-PCA)"
    else:
        Xw = X
        mode_label = f"raw {feature_label} (covariance-PCA)"

    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    mode_label = f"{src_label} | {mode_label}"

    cov = np.cov(Xw, rowvar=False, ddof=1)
    eigvals_asc, eigvecs_asc = np.linalg.eigh(cov)
    eigvals = eigvals_asc[::-1]                     # descending
    eigvecs = eigvecs_asc[:, ::-1]                  # columns are PCs, descending
    # Sign convention: make the largest-|loading| element positive so plots are
    # readable (eigh sign is arbitrary).
    for j in range(eigvecs.shape[1]):
        k = int(np.argmax(np.abs(eigvecs[:, j])))
        if eigvecs[k, j] < 0:
            eigvecs[:, j] *= -1

    labels = token_labels()

    # ------- covariance heatmap --------
    fig, ax = plt.subplots(figsize=(9, 8))
    vmax = float(np.max(np.abs(cov)))
    im = ax.imshow(cov, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(d))
    ax.set_yticks(range(d))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    # Divider between YES (0..11) and NO (12..23) blocks.
    ax.axhline(11.5, color="k", lw=0.5, alpha=0.5)
    ax.axvline(11.5, color="k", lw=0.5, alpha=0.5)
    title_kind = "correlation" if args.standardize else "covariance"
    ax.set_title(
        f"Pregame {feature_label} {title_kind} (t=-10min, {mode_label})\n"
        f"n={n} games, fully-stale dropped"
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    cov_path = PLOTS_DIR / f"covariance{suffix}.png"
    fig.savefig(cov_path, dpi=150)
    plt.close(fig)
    print(f"saved {cov_path}")

    # ------- correlation heatmap (companion) --------
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(d))
    ax.set_yticks(range(d))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axhline(11.5, color="k", lw=0.5, alpha=0.5)
    ax.axvline(11.5, color="k", lw=0.5, alpha=0.5)
    ax.set_title(f"Pregame {feature_label} correlation (t=-10min, {src_label}), n={n} games")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    corr_path = PLOTS_DIR / f"correlation{suffix}.png"
    fig.savefig(corr_path, dpi=150)
    plt.close(fig)
    print(f"saved {corr_path}")

    # ------- scree plot --------
    total = eigvals.sum()
    cum = np.cumsum(eigvals) / total
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(range(1, d + 1), eigvals, marker="o")
    ax1.set_xlabel("component")
    ax1.set_ylabel("eigenvalue")
    ax1.set_yscale("log")
    ax1.set_title("Scree (log scale)")
    ax1.grid(True, alpha=0.3)
    ax2.plot(range(1, d + 1), cum, marker="o")
    ax2.set_xlabel("component")
    ax2.set_ylabel("cumulative variance fraction")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Cumulative variance explained")
    ax2.grid(True, alpha=0.3)
    fig.suptitle(f"Pregame spectrum (t=-10min, {mode_label}), n={n} games")
    fig.tight_layout()
    scree_path = PLOTS_DIR / f"scree{suffix}.png"
    fig.savefig(scree_path, dpi=150)
    plt.close(fig)
    print(f"saved {scree_path}")

    # ------- top-K eigenvectors as bars + heatmap --------
    K = 5
    V = eigvecs[:, :K]
    vmax_v = float(np.max(np.abs(V)))
    fig, axes = plt.subplots(K, 1, figsize=(11, 1.6 * K), sharex=True)
    for j, ax in enumerate(axes):
        colors = ["#c0392b" if v >= 0 else "#2c7fb8" for v in V[:, j]]
        ax.bar(range(d), V[:, j], color=colors)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(11.5, color="k", lw=0.5, alpha=0.5)
        ax.set_ylim(-vmax_v * 1.05, vmax_v * 1.05)
        ax.set_ylabel(f"PC{j+1}\n({eigvals[j]/total*100:.1f}%)", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    axes[-1].set_xticks(range(d))
    axes[-1].set_xticklabels(labels, rotation=90, fontsize=7)
    fig.suptitle(
        f"Top-{K} eigenvectors ({mode_label}), n={n} games"
    )
    fig.tight_layout()
    pcs_path = PLOTS_DIR / f"top_eigenvectors{suffix}.png"
    fig.savefig(pcs_path, dpi=150)
    plt.close(fig)
    print(f"saved {pcs_path}")

    # Companion heatmap (compact comparison view)
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(V.T, cmap="RdBu_r", vmin=-vmax_v, vmax=vmax_v, aspect="auto")
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"PC{j+1} ({eigvals[j]/total*100:.1f}%)" for j in range(K)])
    ax.set_xticks(range(d))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.axvline(11.5, color="k", lw=0.5, alpha=0.5)
    ax.set_title(f"Top-{K} eigenvector loadings ({mode_label})")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    heat_path = PLOTS_DIR / f"top_eigenvectors_heatmap{suffix}.png"
    fig.savefig(heat_path, dpi=150)
    plt.close(fig)
    print(f"saved {heat_path}")

    # ------- text summary --------
    print(f"\nTop {K} eigenvectors (largest |loading| tokens):")
    for j in range(K):
        v = eigvecs[:, j]
        order = np.argsort(-np.abs(v))[:6]
        loadings = "  ".join(f"{labels[i]}:{v[i]:+.2f}" for i in order)
        print(f"  PC{j+1} ({eigvals[j]/total*100:5.2f}%):  {loadings}")

    print("\nTop 8 eigenvalues:")
    for i, lam in enumerate(eigvals[:8]):
        print(f"  PC{i+1:2d}: {lam:.4f}  ({lam/total*100:5.2f}%, cum {cum[i]*100:5.2f}%)")
    print(f"\nTotal variance (trace): {total:.4f}")
    print(f"Effective rank (entropy): {np.exp(-(eigvals/total*np.log(eigvals/total+1e-30)).sum()):.2f}")


if __name__ == "__main__":
    main()
