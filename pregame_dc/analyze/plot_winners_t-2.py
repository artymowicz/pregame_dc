"""Diagnostic plots for moneyline winners at t=-2 min.

For each of 10 games where the strategy fired on a moneyline slot at
t=-120s with edge > 0.05 and the bought side resolved YES, draws a 12-panel
grid (one panel per canonical slot showing the YES side):
  - best YES ask  (x_slot)
  - best YES bid  (= 1 - x_{slot+12})
  - rank-3 PCR predicted probability (model trained at t=-120)
  - vertical dashed red line at t=-120 (fire time)

One PNG per game, written to strategies/pregame_dc/plots/winners_t-2/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

from pregame_dc import paths
from pregame_dc.constants import X_COLS, Y_COLS

OUT_DIR = paths.PACKAGE_ROOT / "plots" / "winners_t-2"
T_FIRE = -120.0
THRESHOLD = 0.05
N_GAMES = 10
PRE_WINDOW_S = 1800   # show last 30 min before kickoff
POST_WINDOW_S = 60    # show 1 min after kickoff for context

SLOT_LABELS = [
    "A_win", "B_win", "Draw",
    "A -1.5", "B -1.5", "A -2.5", "B -2.5",
    "Over 1.5", "Over 2.5", "Over 3.5", "Over 4.5",
    "BTTS",
]


def fit_rank3_at(t_target: float):
    """Fit rank-3 PCR on telonex minus self_collected overlap at t_target seconds.
    Returns mu, sd_safe, beta_K (24, 25)."""
    self_collected_slugs = set(
        pq.read_table(paths.SELF_COLLECTED_LABELED, columns=["game_slug"])
        .to_pandas()["game_slug"].unique()
    )
    df = pq.read_table(
        paths.TELONEX_LABELED,
        columns=["game_slug", "split", "seconds_since_game_start", *X_COLS, *Y_COLS],
    ).to_pandas()
    df = df[df["seconds_since_game_start"] == t_target]
    df = df[~df["game_slug"].isin(self_collected_slugs)]
    X = df[X_COLS].to_numpy(dtype=np.float64)
    y = df[Y_COLS].to_numpy(dtype=np.float64)
    keep = ~(X == 1.0).all(axis=1)
    X_tr = X[keep]
    Y_tr = np.hstack([y[keep], 1.0 - y[keep]])

    n = len(X_tr)
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0, ddof=1)
    sd_safe = np.where(sd > 0, sd, 1.0)
    Z = (X_tr - mu) / sd_safe
    F = np.hstack([Z, np.ones((n, 1))])
    beta = np.linalg.lstsq(F, Y_tr, rcond=None)[0].T
    cov_z = np.cov(Z, rowvar=False, ddof=1)
    eigvals_asc, eigvecs_asc = np.linalg.eigh(cov_z)
    U = eigvecs_asc[:, ::-1]
    for j in range(U.shape[1]):
        k = int(np.argmax(np.abs(U[:, j])))
        if U[k, j] < 0:
            U[:, j] *= -1
    bpc = np.zeros_like(beta[:, :24] @ U)
    bpc[:, :3] = (beta[:, :24] @ U)[:, :3]
    beta_K = np.hstack([bpc @ U.T, beta[:, 24:25]])
    return mu, sd_safe, beta_K, len(X_tr)


def predict(X, mu, sd_safe, beta_K):
    Z = (X - mu) / sd_safe
    F = np.hstack([Z, np.ones((len(Z), 1))])
    return F @ beta_K.T


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"fitting rank-3 PCR at t={T_FIRE}s ...")
    mu, sd_safe, beta_K, n_train = fit_rank3_at(T_FIRE)
    print(f"  trained on {n_train} games")

    # Load self_collected fully (all timestamps) — we need pregame trajectories.
    print(f"loading self_collected dataset ...")
    self_collected = pq.read_table(
        paths.SELF_COLLECTED_LABELED,
        columns=["game_slug", "seconds_since_game_start", *X_COLS, *Y_COLS],
    ).to_pandas()
    print(f"  {len(self_collected):,} rows  {self_collected['game_slug'].nunique()} games")

    # Identify winners at t=-120: moneyline slots {0,1,2}, edge > THRESHOLD,
    # bought outcome resolves YES.
    snap_t = self_collected[self_collected["seconds_since_game_start"] == T_FIRE].copy()
    X_t = snap_t[X_COLS].to_numpy(dtype=np.float64)
    keep = ~(X_t == 1.0).all(axis=1)
    snap_t = snap_t.loc[keep].reset_index(drop=True)
    X_t = X_t[keep]
    pred_t = predict(X_t, mu, sd_safe, beta_K)

    y_t = snap_t[Y_COLS].to_numpy(dtype=np.float64)
    Y_t = np.hstack([y_t, 1.0 - y_t])

    money_slots = [0, 1, 2]
    valid = (X_t >= 0.01) & (X_t <= 0.99)
    edge = pred_t - X_t
    fire = valid & (edge > THRESHOLD)
    fire_money = np.zeros_like(fire)
    fire_money[:, money_slots] = fire[:, money_slots]
    won = fire_money & (Y_t > 0.5)

    winners = []  # (slug, slot, ask, pred, edge)
    for i, row in snap_t.iterrows():
        for s in money_slots:
            if won[i, s]:
                winners.append({
                    "slug": row["game_slug"],
                    "slot": s,
                    "ask": float(X_t[i, s]),
                    "pred": float(pred_t[i, s]),
                    "edge": float(edge[i, s]),
                })
    print(f"  {len(winners)} moneyline-winner fires at t={T_FIRE}s")
    chosen = winners[:N_GAMES]
    print(f"  plotting {len(chosen)} games")

    for w in chosen:
        slug = w["slug"]
        bought_slot = w["slot"]
        g = self_collected[self_collected["game_slug"] == slug].sort_values("seconds_since_game_start")
        ts = g["seconds_since_game_start"].to_numpy(dtype=np.float64)
        X_g = g[X_COLS].to_numpy(dtype=np.float64)
        pred_g = np.clip(predict(X_g, mu, sd_safe, beta_K), 0.0, 1.0)

        # Window
        m = (ts >= -PRE_WINDOW_S) & (ts <= POST_WINDOW_S)
        ts_w = ts[m] / 60.0  # minutes
        X_w = X_g[m]
        pred_w = pred_g[m]

        fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True)
        for s in range(12):
            ax = axes[s // 4][s % 4]
            yes_ask = X_w[:, s]                  # x_s
            yes_bid = 1.0 - X_w[:, s + 12]       # 1 - x_{s+12}
            pred_y  = pred_w[:, s]
            ax.plot(ts_w, yes_ask, color="#d62728", lw=1.0, label="YES ask")
            ax.plot(ts_w, yes_bid, color="#2ca02c", lw=1.0, label="YES bid")
            ax.plot(ts_w, pred_y, color="#1f77b4", lw=1.5, label="NN pred")
            ax.axvline(T_FIRE / 60.0, color="red", ls="--", lw=1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"slot {s}: {SLOT_LABELS[s]}"
                         + ("  ← BOUGHT" if s == bought_slot else ""),
                         fontsize=9)
            ax.grid(alpha=0.3)
            if s == 0:
                ax.legend(loc="best", fontsize=7)
        for ax in axes[-1]:
            ax.set_xlabel("min from kickoff")
        title = (f"{slug}  (winner: slot {bought_slot} {SLOT_LABELS[bought_slot]}, "
                 f"ask={w['ask']:.3f}, pred={w['pred']:.3f}, edge=+{w['edge']:.3f})")
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = OUT_DIR / f"{slug}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  wrote {out}")

    print(f"\nall plots in {OUT_DIR}/")


if __name__ == "__main__":
    main()
