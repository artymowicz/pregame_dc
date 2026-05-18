"""Diagnostic plot: moneyline fires at edge>0.10, SC vs TX time series.

For each (game, moneyline-slot) where the rank-3 PCR model fires
(pred − ask > 0.10) at t = −10 min in EITHER self_collected or telonex,
plot the time series of ask, bid (=1 − complement-ask), and model
predicted p from both data sources, on the same axes.
"""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

from pregame_dc import paths
from pregame_dc.constants import X_COLS, Y_COLS, MARKET_LABELS

T_FIRE = -600.0
EDGE_THR = 0.10
ASK_LO, ASK_HI = 0.01, 0.99
KEEP_TOP = 3
MONEYLINE_SLOTS = [0, 1, 2, 12, 13, 14]
T_LO, T_HI = -1800.0, 600.0

OUT_DIR = paths.PACKAGE_ROOT / "plots" / "moneyline_edge_gt_0.10"


def load_at_t(path, t, exclude_slugs=None):
    df = pq.read_table(
        path,
        columns=["game_slug", "seconds_since_game_start", *X_COLS, *Y_COLS],
    ).to_pandas()
    df = df[df["seconds_since_game_start"] == t]
    if exclude_slugs is not None:
        df = df[~df["game_slug"].isin(exclude_slugs)]
    X = df[X_COLS].to_numpy(dtype=np.float64)
    y = df[Y_COLS].to_numpy(dtype=np.float64)
    keep = ~(X == 1.0).all(axis=1)
    Y = np.hstack([y[keep], 1.0 - y[keep]])
    slugs = df.loc[keep, "game_slug"].to_numpy()
    return X[keep], Y, slugs


def fit_topk(X_tr, Y_tr, K):
    n_tr = len(X_tr)
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0, ddof=1)
    sd_safe = np.where(sd > 0, sd, 1.0)
    Z_tr = (X_tr - mu) / sd_safe
    F_tr = np.hstack([Z_tr, np.ones((n_tr, 1))])
    beta = np.linalg.lstsq(F_tr, Y_tr, rcond=None)[0].T
    cov_z = np.cov(Z_tr, rowvar=False, ddof=1)
    eigvals_asc, eigvecs_asc = np.linalg.eigh(cov_z)
    U = eigvecs_asc[:, ::-1]
    for j in range(U.shape[1]):
        k = int(np.argmax(np.abs(U[:, j])))
        if U[k, j] < 0:
            U[:, j] *= -1
    bpc = np.zeros_like(beta[:, :24] @ U)
    bpc[:, :K] = (beta[:, :24] @ U)[:, :K]
    beta_K = np.hstack([bpc @ U.T, beta[:, 24:25]])

    def predict(X_eval):
        Z = (X_eval - mu) / sd_safe
        F = np.hstack([Z, np.ones((len(Z), 1))])
        return F @ beta_K.T
    return predict


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sc_set = set(pq.read_table(paths.SELF_COLLECTED_LABELED, columns=["game_slug"])
                 .to_pandas()["game_slug"].unique())
    tx_set = set(pq.read_table(paths.TELONEX_LABELED, columns=["game_slug"])
                 .to_pandas()["game_slug"].unique())
    overlap = sc_set & tx_set

    # Train at fire time on telonex MINUS self_collected slugs.
    X_tr, Y_tr, _ = load_at_t(paths.TELONEX_LABELED, T_FIRE, exclude_slugs=sc_set)
    print(f"train n={len(X_tr)} at t={T_FIRE:+.0f}s")
    predict = fit_topk(X_tr, Y_tr, KEEP_TOP)

    # Eval matrices at fire time on overlap, both sources.
    X_sc_t, _, slugs_sc = load_at_t(
        paths.SELF_COLLECTED_LABELED, T_FIRE, exclude_slugs=(sc_set - overlap)
    )
    X_tx_t, _, slugs_tx = load_at_t(
        paths.TELONEX_LABELED, T_FIRE, exclude_slugs=(tx_set - overlap)
    )
    pred_sc = predict(X_sc_t)
    pred_tx = predict(X_tx_t)

    fires = set()
    for slugs, X, P in [(slugs_sc, X_sc_t, pred_sc), (slugs_tx, X_tx_t, pred_tx)]:
        edge = P - X
        valid = (X >= ASK_LO) & (X <= ASK_HI)
        for r in range(len(slugs)):
            for s in MONEYLINE_SLOTS:
                if valid[r, s] and edge[r, s] > EDGE_THR:
                    fires.add((slugs[r], s))
    fires = sorted(fires)
    print(f"unique (game, slot) fires (moneyline, edge>{EDGE_THR}, t=-10min): {len(fires)}")

    needed = sorted({slug for slug, _ in fires})
    sc_full = pq.read_table(
        paths.SELF_COLLECTED_LABELED,
        columns=["game_slug", "seconds_since_game_start", *X_COLS],
    ).to_pandas()
    tx_full = pq.read_table(
        paths.TELONEX_LABELED,
        columns=["game_slug", "seconds_since_game_start", *X_COLS],
    ).to_pandas()
    sc_full = sc_full[sc_full["game_slug"].isin(needed)]
    tx_full = tx_full[tx_full["game_slug"].isin(needed)]

    n = len(fires)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.0 * nrows), squeeze=False)

    sc_color, tx_color = "#1f77b4", "#d62728"

    for idx, (slug, slot) in enumerate(fires):
        ax = axes[idx // ncols][idx % ncols]
        comp_slot = slot + 12 if slot < 12 else slot - 12
        side = "YES" if slot < 12 else "NO"
        market = MARKET_LABELS[slot if slot < 12 else slot - 12]

        for ds_name, df, color in [("SC", sc_full, sc_color), ("TX", tx_full, tx_color)]:
            sub = df[df["game_slug"] == slug].sort_values("seconds_since_game_start")
            if len(sub) == 0:
                continue
            t = sub["seconds_since_game_start"].to_numpy()
            X_all = sub[X_COLS].to_numpy(dtype=np.float64)

            ask = X_all[:, slot]
            comp_ask = X_all[:, comp_slot]
            bid = 1.0 - comp_ask

            row_valid = ~(X_all == 1.0).all(axis=1)
            preds = np.full(len(X_all), np.nan)
            if row_valid.any():
                preds[row_valid] = predict(X_all[row_valid])[:, slot]

            ask_plot = np.where(ask < 1.0, ask, np.nan)
            bid_plot = np.where(comp_ask < 1.0, bid, np.nan)

            tmask = (t >= T_LO) & (t <= T_HI)
            tmin = t[tmask] / 60.0
            ax.plot(tmin, ask_plot[tmask], color=color, lw=1.4, label=f"{ds_name} ask")
            ax.plot(tmin, bid_plot[tmask], color=color, lw=1.0, ls=":", label=f"{ds_name} bid")
            ax.plot(tmin, preds[tmask], color=color, lw=1.6, ls="--", alpha=0.6,
                    label=f"{ds_name} pred p")

        ax.axvline(-10, color="black", lw=0.8, ls="--")
        ax.set_xlabel("minutes to kickoff")
        ax.set_ylabel("price")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{slug}\n{market} [{side}]  (slot {slot})", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best", ncol=2)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        f"Moneyline edge>{EDGE_THR} fires at t=-10 min  |  SC (blue) vs TX (red)\n"
        f"solid=ask, dotted=bid (=1−complement ask), dashed=model pred",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / "moneyline_edge_gt_0.10.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
