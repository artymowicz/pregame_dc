"""Train impute-only rank-K PCR on telonex (excluding self_collected overlap)
and save model parameters.

Differs from fit_save_model.py by replacing 1.0 placeholder cells with
per-column means (computed over non-sentinel rows of the training set)
*before* fitting mu/sd/PCA/OLS. The saved `impute_values` (shape (24,))
must be applied at inference: replace any ask of 1.0 with `impute_values[i]`
before standardising with mu/sd_safe.

Saved keys:
    mu, sd_safe, beta_K, U, eigvals      — standard PCR artifacts (24-dim)
    impute_values                        — (24,) per-column mean for 1.0
    T_TARGET, K, train_n                 — provenance
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pregame_dc.constants import X_COLS, Y_COLS
from pregame_dc import paths

PLACEHOLDER = 1.0


def load_telonex_excluding_self_collected(t_target: float):
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
    keep = ~(X == PLACEHOLDER).all(axis=1)
    X = X[keep]
    y = y[keep]
    Y = np.hstack([y, 1.0 - y])
    return X, Y


def column_means_excluding_sentinel(X):
    is_real = X != PLACEHOLDER
    sums = np.where(is_real, X, 0.0).sum(axis=0)
    counts = is_real.sum(axis=0)
    fallback = X[is_real].mean() if is_real.any() else 0.5
    return np.where(counts > 0, sums / np.maximum(counts, 1), fallback)


def replace_sentinels(X, impute_values):
    return np.where(X == PLACEHOLDER, impute_values[None, :], X)


def fit_rank_k(X_tr: np.ndarray, Y_tr: np.ndarray, K: int):
    """Returns mu, sd_safe, beta_K, U, eigvals. X_tr is assumed to already
    have sentinel cells imputed."""
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
                    help="seconds_since_game_start to train at "
                         "(default -600 = -10min)")
    ap.add_argument("--out", type=Path, required=True,
                    help="output .npz path (e.g. pregame_dc/models/rank3_t-10min_imp.npz)")
    ap.add_argument("--k", type=int, default=3, help="rank truncation (default 3)")
    args = ap.parse_args()

    print(f"loading telonex (excluding self_collected overlap) at "
          f"t={args.time_seconds}s ...")
    X_raw, Y_tr = load_telonex_excluding_self_collected(args.time_seconds)
    sent_frac = float((X_raw == PLACEHOLDER).mean())
    print(f"  train n = {len(X_raw)} games, sentinel cells = {sent_frac:.2%}")

    impute_values = column_means_excluding_sentinel(X_raw)
    X_tr = replace_sentinels(X_raw, impute_values)
    print(f"  impute_values range: [{impute_values.min():.3f}, "
          f"{impute_values.max():.3f}],  mean={impute_values.mean():.3f}")

    mu, sd_safe, beta_K, U, eigvals = fit_rank_k(X_tr, Y_tr, args.k)
    print(f"  mu shape: {mu.shape}, sd_safe shape: {sd_safe.shape}")
    print(f"  beta_K shape: {beta_K.shape} (24 outcomes x 25 features-+-bias)")
    print(f"  top-5 eigvals: {[f'{v:.3f}' for v in eigvals[:5]]}")
    print(f"  top-{args.k} cumvar: "
          f"{eigvals[:args.k].sum() / eigvals.sum() * 100:.1f}% of trace")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        mu=mu,
        sd_safe=sd_safe,
        beta_K=beta_K,
        U=U,
        eigvals=eigvals,
        impute_values=impute_values,
        T_TARGET=np.array(args.time_seconds),
        K=np.array(args.k),
        train_n=np.array(len(X_tr)),
    )
    print(f"\nsaved to {args.out}")
    print("\nINFERENCE NOTE:")
    print("  This model was fit on imputed inputs. At inference, before")
    print("  computing Z = (x - mu)/sd_safe, replace any ask cell equal to")
    print("  1.0 with the corresponding `impute_values[i]`.")


if __name__ == "__main__":
    main()
