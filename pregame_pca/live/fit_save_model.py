"""Train rank-3 PCR on telonex (excluding self_collected overlap) and save model parameters.

The live bot loads `mu`, `sd_safe`, `beta_K` from the resulting .npz and uses
them as fixed inference constants. `U` and `eigvals` are saved for diagnostics
but are not required at inference time.

Outcomes are read from the labeled parquet's `y_0..y_11` columns directly;
the 24-element outcome vector consumed by the regression is reconstructed via
`Y = hstack([y, 1 - y])`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pregame_pca.constants import X_COLS, Y_COLS
from pregame_pca import paths


def load_telonex_excluding_self_collected(t_target: float):
    """Returns (X, Y) arrays restricted to the t-per_game_data of telonex games not
    present in self_collected. X has 24 ask columns, Y has 24 outcome columns
    (12 YES followed by 12 NO = 1 - YES)."""
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
    ap.add_argument("--time-seconds", type=float, default=-1500.0,
                    help="seconds_since_game_start to train at (default -1500 = -25min)")
    ap.add_argument("--out", type=Path, default=paths.MODEL_T_25MIN)
    ap.add_argument("--k", type=int, default=3, help="rank truncation (default 3)")
    args = ap.parse_args()

    print(f"loading telonex (excluding self_collected overlap) at t={args.time_seconds}s ...")
    X_tr, Y_tr = load_telonex_excluding_self_collected(args.time_seconds)
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
