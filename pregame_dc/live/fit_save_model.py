"""Train rank-K PCR on telonex and save model parameters.

The live bot loads `mu`, `sd_safe`, `beta_K`, `U` from the resulting .npz.
`U` is required at inference time for computing res_norm (the live firing
gate); `eigvals` is saved for diagnostics.

By default trains on the full telonex set at the requested timepoint. The
`--exclude-self-collected` flag keeps the historical behaviour of dropping
games that also appear in the self_collected dataset; we leave the rest of
telonex (a strict superset under our preferred data-source) as training data.

Outcomes are read from the labeled parquet's `y_0..y_11` columns directly;
the 24-element outcome vector consumed by the regression is reconstructed via
`Y = hstack([y, 1 - y])`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pregame_dc.constants import X_COLS, Y_COLS
from pregame_dc import paths


def load_telonex(t_target: float, exclude_self_collected: bool = False):
    """Returns (X, Y) arrays at the given t-offset on the telonex labeled
    dataset. X has 24 ask columns, Y has 24 outcome columns (12 YES followed
    by 12 NO = 1 - YES). When `exclude_self_collected` is True (legacy),
    games that also appear in the self_collected dataset are dropped."""
    df = pq.read_table(
        paths.TELONEX_LABELED,
        columns=["game_slug", "split", "seconds_since_game_start", *X_COLS, *Y_COLS],
    ).to_pandas()
    df = df[df["seconds_since_game_start"] == t_target]
    if exclude_self_collected:
        self_collected_slugs = set(
            pq.read_table(paths.SELF_COLLECTED_LABELED, columns=["game_slug"])
            .to_pandas()["game_slug"].unique()
        )
        df = df[~df["game_slug"].isin(self_collected_slugs)]
    X = df[X_COLS].to_numpy(dtype=np.float64)
    y = df[Y_COLS].to_numpy(dtype=np.float64)
    keep = ~(X == 1.0).all(axis=1)
    X = X[keep]
    y = y[keep]
    Y = np.hstack([y, 1.0 - y])
    return X, Y


def fit_rank_k(X_tr: np.ndarray, Y_tr: np.ndarray, K: int):
    """Returns mu, sd_safe, beta_K, U, eigvals."""
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0, ddof=1)
    sd_safe = np.where(sd > 0, sd, 1.0)
    Z_tr = (X_tr - mu) / sd_safe
    n_tr = len(X_tr)
    F_tr = np.hstack([Z_tr, np.ones((n_tr, 1))])
    beta_T, *_ = np.linalg.lstsq(F_tr, Y_tr, rcond=None)
    beta = beta_T.T   # (24, 25)

    cov_z = np.cov(Z_tr, rowvar=False, ddof=1)
    eigvals_asc, eigvecs_asc = np.linalg.eigh(cov_z)
    eigvals = eigvals_asc[::-1]
    U = eigvecs_asc[:, ::-1]
    for j in range(U.shape[1]):
        k = int(np.argmax(np.abs(U[:, j])))
        if U[k, j] < 0:
            U[:, j] *= -1

    beta_pc = beta[:, :24] @ U
    bpc = np.zeros_like(beta_pc)
    bpc[:, :K] = beta_pc[:, :K]
    beta_K = np.hstack([bpc @ U.T, beta[:, 24:25]])

    return mu, sd_safe, beta_K, U, eigvals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-seconds", type=float, default=-600.0,
                    help="seconds_since_game_start to train at (default -600 = -10min)")
    ap.add_argument("--out", type=Path, default=paths.MODEL_T_10MIN_K4)
    ap.add_argument("--k", type=int, default=4, help="rank truncation (default 4)")
    ap.add_argument("--exclude-self-collected", action="store_true",
                    help="Drop games that also appear in self_collected from the "
                         "training set (legacy; off by default — sc games stay in).")
    args = ap.parse_args()

    msg = "loading telonex"
    if args.exclude_self_collected:
        msg += " (excluding self_collected overlap)"
    print(f"{msg} at t={args.time_seconds}s ...")
    X_tr, Y_tr = load_telonex(args.time_seconds, exclude_self_collected=args.exclude_self_collected)
    print(f"  train n = {len(X_tr)} games")

    mu, sd_safe, beta_K, U, eigvals = fit_rank_k(X_tr, Y_tr, args.k)
    print(f"  mu shape: {mu.shape}, sd_safe shape: {sd_safe.shape}")
    print(f"  beta_K shape: {beta_K.shape} (24 outcomes x 25 features-+-bias)")
    print(f"  top-5 eigvals: {[f'{v:.3f}' for v in eigvals[:5]]}")
    print(f"  top-3 cumvar: {eigvals[:3].sum() / eigvals.sum() * 100:.1f}% of trace")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        mu=mu,
        sd_safe=sd_safe,
        beta_K=beta_K,
        U=U,
        eigvals=eigvals,
        T_TARGET=np.array(args.time_seconds),
        K=np.array(args.k),
        train_n=np.array(len(X_tr)),
    )
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
