"""Head-to-head OOS eval of vanilla vs sentinel-imputation rank-K PCR.

Train: telonex games not present in self_collected.
Eval:  self_collected.

For each t in T_TARGETS, fits both:
  - VANILLA: standardise raw X (with 1.0 placeholders left in place),
    PCR to rank K, OLS the projected features against Y.
  - IMPUTE: replace any x == 1.0 with the per-column mean over non-sentinel
    rows of the training set, then fit standard PCR rank K.

For each model reports Brier R² over the 12 YES marginals plus a
threshold-sweep PnL/trade table by market type.
"""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq

from pregame_dc import paths
from pregame_dc.constants import X_COLS, Y_COLS, TYPE_FOR_SLOT

ASK_LO, ASK_HI = 0.01, 0.99
THRESHOLDS = [0.00, 0.02, 0.05, 0.10, 0.15]
KEEP_TOP = 3
PLACEHOLDER = 1.0

T_TARGETS = [-1500.0, -600.0, -120.0]   # seconds: -25, -10, -2 min


def load(path, t_target, exclude_slugs=None):
    df = pq.read_table(
        path,
        columns=["game_slug", "seconds_since_game_start", *X_COLS, *Y_COLS],
    ).to_pandas()
    df = df[df["seconds_since_game_start"] == t_target]
    if exclude_slugs is not None:
        df = df[~df["game_slug"].isin(exclude_slugs)]
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


def fit_topk(X_tr, Y_tr, K):
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0, ddof=1)
    sd_safe = np.where(sd > 0, sd, 1.0)
    Z_tr = (X_tr - mu) / sd_safe
    n_tr = len(X_tr)
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
    return mu, sd_safe, beta_K


def predict(X_eval, mu, sd_safe, beta_K):
    Z = (X_eval - mu) / sd_safe
    F = np.hstack([Z, np.ones((len(Z), 1))])
    return F @ beta_K.T


def metrics(yhat, Y_va, ask, Y_tr_mean):
    yhat = np.clip(yhat, 0.0, 1.0)
    yes_idx = list(range(12))
    base_yhat = np.tile(Y_tr_mean, (len(Y_va), 1))
    base_mse = ((base_yhat - Y_va) ** 2).mean(axis=0)
    mse = ((yhat - Y_va) ** 2).mean(axis=0)
    ss_tot = ((Y_va - Y_va.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ((yhat - Y_va) ** 2).sum(axis=0) / np.where(ss_tot > 0, ss_tot, 1.0)
    return {
        "base_brier_yes": float(base_mse[yes_idx].mean()),
        "brier_yes": float(mse[yes_idx].mean()),
        "r2_yes": float(r2[yes_idx].mean()),
        "yhat": yhat,
    }


def pnl_table(yhat, Y_va, ask):
    edge = yhat - ask
    pnl = Y_va - ask
    valid = (ask >= ASK_LO) & (ask <= ASK_HI)
    type_to_slots = {}
    for slot, mt in TYPE_FOR_SLOT.items():
        for s in (slot, slot + 12):
            type_to_slots.setdefault(mt, []).append(s)

    rows = []
    for thr in THRESHOLDS:
        fire = valid & (edge > thr)
        cells = []
        n_all = int(fire.sum())
        cells.append(f"{n_all} / {pnl[fire].mean():+.4f}" if n_all else "0 / n/a")
        for mt in ("moneyline", "spread", "totals", "btts"):
            slots = type_to_slots[mt]
            mask = np.zeros_like(fire)
            mask[:, slots] = fire[:, slots]
            n = int(mask.sum())
            cells.append(f"{n} / {pnl[mask].mean():+.4f}" if n else "0 / n/a")
        rows.append((thr, cells))
    return rows


def print_pnl(title, rows):
    print(f"\n  {title}")
    print(f"    {'thresh':>7s}  {'agg n/$':>16s}  {'mny n/$':>16s}  "
          f"{'spr n/$':>16s}  {'tot n/$':>16s}  {'btts n/$':>16s}")
    for thr, cells in rows:
        print(f"    {thr:>7.2f}  " + "  ".join(c.rjust(16) for c in cells))


def main():
    self_collected_slugs = set(
        pq.read_table(paths.SELF_COLLECTED_LABELED, columns=["game_slug"])
        .to_pandas()["game_slug"].unique()
    )

    for T in T_TARGETS:
        X_tr_raw, Y_tr = load(paths.TELONEX_LABELED, T,
                              exclude_slugs=self_collected_slugs)
        X_va, Y_va = load(paths.SELF_COLLECTED_LABELED, T, exclude_slugs=None)
        sent_frac_tr = float((X_tr_raw == PLACEHOLDER).mean())
        sent_frac_va = float((X_va == PLACEHOLDER).mean())

        Y_tr_mean = Y_tr.mean(axis=0)

        # Vanilla: raw asks, sentinels left in place.
        mu_v, sd_v, beta_v = fit_topk(X_tr_raw, Y_tr, KEEP_TOP)
        yhat_v = predict(X_va, mu_v, sd_v, beta_v)
        m_v = metrics(yhat_v, Y_va, X_va, Y_tr_mean)

        # Imputed: replace 1.0 cells with per-col mean of non-sentinel rows.
        impute_values = column_means_excluding_sentinel(X_tr_raw)
        X_tr_imp = replace_sentinels(X_tr_raw, impute_values)
        X_va_imp = replace_sentinels(X_va, impute_values)
        mu_i, sd_i, beta_i = fit_topk(X_tr_imp, Y_tr, KEEP_TOP)
        yhat_i = predict(X_va_imp, mu_i, sd_i, beta_i)
        m_i = metrics(yhat_i, Y_va, X_va, Y_tr_mean)

        print(f"\n{'='*78}")
        print(f"t = {int(T)}s ({T/60:+.1f} min)   "
              f"train n={len(Y_tr)}  val n={len(Y_va)}   "
              f"sentinel frac: train={sent_frac_tr:.2%} val={sent_frac_va:.2%}")
        print('='*78)
        print(f"  Baseline Brier (Y_tr mean): {m_v['base_brier_yes']:.4f}")
        print(f"  {'model':<10s}  {'Brier(YES)':>10s}  {'R²(YES)':>10s}")
        print(f"  {'vanilla':<10s}  {m_v['brier_yes']:>10.4f}  {m_v['r2_yes']:>+10.4f}")
        print(f"  {'impute':<10s}  {m_i['brier_yes']:>10.4f}  {m_i['r2_yes']:>+10.4f}")

        # PnL tables: ask is the raw self-collected ask (no imputation for trading).
        print_pnl("vanilla — PnL/trade by market type", pnl_table(m_v["yhat"], Y_va, X_va))
        print_pnl("impute  — PnL/trade by market type", pnl_table(m_i["yhat"], Y_va, X_va))


if __name__ == "__main__":
    main()
