"""Per-cell ask comparison: self_collected vs telonex.

Goal: separate "coverage gap" (only one source has a quote at this
(game, t, slot)) from "data quality gap" (both have a quote but they
disagree). Restricted to the overlap of game slugs.

Caveat (per docs/findings.md): SC was excluded from the latest model
run because of a known pipeline bug being fixed in parallel. Disagreements
measured here may partially reflect that bug rather than a steady-state
property of the SC feed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

from pregame_dc import paths
from pregame_dc.constants import X_COLS, TYPE_FOR_SLOT, MARKET_LABELS

OUT_DIR = paths.PACKAGE_ROOT / "plots" / "sc_vs_tx_per_cell"
PLACEHOLDER = 1.0
MARKET_TYPES = ["moneyline", "spread", "totals", "btts"]
T_BUCKETS = [
    ("pre-game-far",  -np.inf, -600.0),   # t ≤ −10min (note: ≤ via boundary handling below)
    ("pre-game-near", -600.0,     0.0),   # −10min < t ≤ 0
    ("live",             0.0,  np.inf),   # t > 0
]


def slot_market_type(slot: int) -> str:
    return TYPE_FOR_SLOT[slot if slot < 12 else slot - 12]


def slot_side(slot: int) -> str:
    return "YES" if slot < 12 else "NO"


def load_long():
    """Inner-merge SC and TX on (game_slug, t), restricted to overlap slugs.
    Returns long-format DataFrame: one row per (game_slug, t, slot)."""
    sc = pq.read_table(
        paths.SELF_COLLECTED_LABELED,
        columns=["game_slug", "seconds_since_game_start", *X_COLS],
    ).to_pandas()
    tx = pq.read_table(
        paths.TELONEX_LABELED,
        columns=["game_slug", "seconds_since_game_start", *X_COLS],
    ).to_pandas()

    overlap = set(sc["game_slug"].unique()) & set(tx["game_slug"].unique())
    sc = sc[sc["game_slug"].isin(overlap)]
    tx = tx[tx["game_slug"].isin(overlap)]

    merged = sc.merge(
        tx, on=["game_slug", "seconds_since_game_start"], suffixes=("_sc", "_tx")
    )
    sc_cols = [f"{c}_sc" for c in X_COLS]
    tx_cols = [f"{c}_tx" for c in X_COLS]
    sc_arr = merged[sc_cols].to_numpy(dtype=np.float64)
    tx_arr = merged[tx_cols].to_numpy(dtype=np.float64)
    n_rows = len(merged)
    n_slots = 24

    long = pd.DataFrame({
        "game_slug": np.repeat(merged["game_slug"].to_numpy(), n_slots),
        "t":         np.repeat(merged["seconds_since_game_start"].to_numpy(), n_slots),
        "slot":      np.tile(np.arange(n_slots), n_rows),
        "ask_sc":    sc_arr.reshape(-1),
        "ask_tx":    tx_arr.reshape(-1),
    })
    long["market"] = long["slot"].map(slot_market_type)
    long["side"]   = long["slot"].map(slot_side)
    return long, len(overlap), n_rows


def classify(df: pd.DataFrame) -> pd.Series:
    """BOTH / SC_only / TX_only / NEITHER."""
    sc_present = df["ask_sc"] < PLACEHOLDER
    tx_present = df["ask_tx"] < PLACEHOLDER
    out = np.full(len(df), "NEITHER", dtype=object)
    out[sc_present & tx_present] = "BOTH"
    out[sc_present & ~tx_present] = "SC_only"
    out[~sc_present & tx_present] = "TX_only"
    return pd.Series(out, index=df.index, name="status")


def t_bucket(t: pd.Series) -> pd.Series:
    out = pd.Series(np.full(len(t), "live", dtype=object), index=t.index)
    out[t <= -600.0] = "pre-game-far"
    out[(t > -600.0) & (t <= 0.0)] = "pre-game-near"
    return out


def coverage_table(df: pd.DataFrame, group_label: str, group_vals) -> str:
    """Return a printable table: one row per group_val, columns by status."""
    lines = []
    header = f"{group_label:<14s}  {'N':>10s}  {'BOTH':>10s}  {'SC_only':>10s}  {'TX_only':>10s}  {'NEITHER':>10s}"
    lines.append(header)
    lines.append("-" * len(header))
    for g in group_vals:
        sub = df if g == "ALL" else df[df[group_label] == g]
        if len(sub) == 0:
            lines.append(f"{g:<14s}  {'0':>10s}")
            continue
        n = len(sub)
        counts = sub["status"].value_counts()
        cells = [counts.get(s, 0) for s in ("BOTH", "SC_only", "TX_only", "NEITHER")]
        pcts = [c / n * 100 for c in cells]
        line = f"{g:<14s}  {n:>10,d}"
        for c, p in zip(cells, pcts):
            line += f"  {c:>5,d} {p:>4.1f}%"
        lines.append(line)
    return "\n".join(lines)


def agreement_table(both: pd.DataFrame, group_label: str, group_vals) -> str:
    lines = []
    header = (f"{group_label:<14s}  {'N_both':>9s}  {'mean diff':>9s}  "
              f"{'p50|d|':>7s}  {'p90|d|':>7s}  {'p95|d|':>7s}  {'p99|d|':>7s}  "
              f"{'<0.5¢':>6s}  {'<1¢':>6s}  {'<2¢':>6s}  {'<5¢':>6s}")
    lines.append(header)
    lines.append("-" * len(header))
    for g in group_vals:
        sub = both if g == "ALL" else both[both[group_label] == g]
        n = len(sub)
        if n == 0:
            lines.append(f"{g:<14s}  {'0':>9s}")
            continue
        diff = sub["ask_sc"].to_numpy() - sub["ask_tx"].to_numpy()
        adiff = np.abs(diff)
        line = (
            f"{g:<14s}  {n:>9,d}  {diff.mean():>+9.4f}  "
            f"{np.percentile(adiff, 50):>7.4f}  "
            f"{np.percentile(adiff, 90):>7.4f}  "
            f"{np.percentile(adiff, 95):>7.4f}  "
            f"{np.percentile(adiff, 99):>7.4f}  "
            f"{(adiff < 0.005).mean()*100:>5.1f}%  "
            f"{(adiff < 0.01).mean()*100:>5.1f}%  "
            f"{(adiff < 0.02).mean()*100:>5.1f}%  "
            f"{(adiff < 0.05).mean()*100:>5.1f}%"
        )
        lines.append(line)
    return "\n".join(lines)


def plot_coverage_over_time(df: pd.DataFrame, out_path):
    grp = df.groupby(["t", "status"]).size().unstack(fill_value=0)
    for s in ("BOTH", "SC_only", "TX_only", "NEITHER"):
        if s not in grp.columns:
            grp[s] = 0
    pct = grp.div(grp.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(12, 5))
    tmin = pct.index.to_numpy() / 60.0
    for status, color in [("BOTH", "#2ca02c"), ("SC_only", "#1f77b4"),
                          ("TX_only", "#d62728"), ("NEITHER", "#7f7f7f")]:
        ax.plot(tmin, pct[status].to_numpy(), label=status, color=color, lw=1.2)
    ax.axvline(-10, color="black", lw=0.8, ls="--", label="t=−10 min")
    ax.axvline(0, color="black", lw=0.8, ls=":", label="kickoff")
    ax.set_xlabel("minutes to kickoff")
    ax.set_ylabel("% of (game, slot) cells")
    ax.set_title("Coverage status over time (overlap games, all 24 slots)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_abs_diff_hist(both: pd.DataFrame, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False)
    for ax, mt in zip(axes.flat, MARKET_TYPES):
        sub = both[both["market"] == mt]
        if len(sub) == 0:
            ax.set_title(f"{mt} (no data)")
            continue
        adiff = np.abs(sub["ask_sc"].to_numpy() - sub["ask_tx"].to_numpy())
        ax.hist(adiff, bins=np.linspace(0, 0.2, 81), color="#1f77b4", alpha=0.8)
        for q, lbl, color in [(50, "p50", "#2ca02c"), (95, "p95", "#ff7f0e"),
                              (99, "p99", "#d62728")]:
            v = np.percentile(adiff, q)
            ax.axvline(v, color=color, lw=1.0, ls="--", label=f"{lbl}={v:.4f}")
        ax.set_xlabel("|ask_sc − ask_tx|")
        ax.set_ylabel("count")
        ax.set_title(f"{mt}  (n={len(sub):,} BOTH cells, x clipped to 0.20)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Per-cell ask disagreement on BOTH cells", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_scatter(both: pd.DataFrame, out_path, max_pts=200_000):
    fig, ax = plt.subplots(figsize=(8, 8))
    if len(both) > max_pts:
        sample = both.sample(max_pts, random_state=0)
    else:
        sample = both
    colors = {"moneyline": "#1f77b4", "spread": "#ff7f0e",
              "totals": "#2ca02c", "btts": "#d62728"}
    for mt in MARKET_TYPES:
        sub = sample[sample["market"] == mt]
        if len(sub) == 0:
            continue
        ax.scatter(sub["ask_sc"], sub["ask_tx"], s=2, alpha=0.25,
                   color=colors[mt], label=f"{mt} (n={len(sub):,} shown)")
    ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--", label="y=x")
    ax.set_xlabel("ask_sc")
    ax.set_ylabel("ask_tx")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(f"ask_sc vs ask_tx on BOTH cells "
                 f"(showing {min(len(both), max_pts):,} of {len(both):,})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    long, n_overlap, n_rows = load_long()
    long["status"] = classify(long)
    long["t_bucket"] = t_bucket(long["t"])

    print(f"overlap games: {n_overlap}")
    print(f"merged (game, t) rows: {n_rows:,}")
    print(f"total cells: {len(long):,}  (= {n_rows} × 24)")
    print()

    print("=" * 80)
    print("COVERAGE — overall + by market type + by time window")
    print("=" * 80)
    print(coverage_table(long, "ALL", ["ALL"]))
    print()
    print(coverage_table(long, "market", MARKET_TYPES))
    print()
    print(coverage_table(long, "side", ["YES", "NO"]))
    print()
    print(coverage_table(long, "t_bucket", ["pre-game-far", "pre-game-near", "live"]))
    print()

    both = long[long["status"] == "BOTH"]

    print("=" * 80)
    print("AGREEMENT on BOTH cells")
    print("=" * 80)
    print(agreement_table(both, "ALL", ["ALL"]))
    print()
    print(agreement_table(both, "market", MARKET_TYPES))
    print()
    print(agreement_table(both, "side", ["YES", "NO"]))
    print()
    print(agreement_table(both, "t_bucket", ["pre-game-far", "pre-game-near", "live"]))
    print()

    # Cross-check against the 7-fire sample from earlier turn.
    cross_slugs = ["nor-kbk-bog", "por-cas-ben"]
    sub = both[(both["game_slug"].isin(cross_slugs)) & (both["t"] == -600.0)
               & (both["slot"].isin([0, 1, 2, 12, 13, 14]))]
    print("Spot-check vs earlier moneyline edge>0.10 fires (expected: 3 BOTH cells, |diff|=0):")
    if len(sub) == 0:
        print("  no BOTH cells matched")
    for _, r in sub.iterrows():
        d = r["ask_sc"] - r["ask_tx"]
        print(f"  {r['game_slug']:<14s} slot={int(r['slot']):>2d}  "
              f"sc={r['ask_sc']:.3f}  tx={r['ask_tx']:.3f}  diff={d:+.4f}")
    print()

    plot_coverage_over_time(long, OUT_DIR / "coverage_over_time.png")
    plot_abs_diff_hist(both, OUT_DIR / "abs_diff_hist.png")
    plot_scatter(both, OUT_DIR / "scatter_sc_vs_tx.png")
    print(f"plots written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
