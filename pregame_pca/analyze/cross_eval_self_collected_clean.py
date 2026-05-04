"""Clean cross-eval: train on telonex games NOT in self_collected, eval on self_collected.

96.5% of self_collected games overlap with telonex (same Polymarket games, different data
sources). This script trains only on the non-overlapping telonex games so the
self_collected evaluation is truly held out by game identity. With ~2182 train games and
72 effective parameters in the rank-3 PCR, overfitting risk is intrinsically
low; this script just confirms the leakage isn't driving conclusions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pregame_pca import paths
from pregame_pca.constants import X_COLS, Y_COLS, TYPE_FOR_SLOT

T_TARGET = -600.0
ASK_LO, ASK_HI = 0.01, 0.99
THRESHOLDS = [0.00, 0.02, 0.05, 0.10, 0.15]
KEEP_TOP = 3


def load(path: Path, exclude_slugs=None):
    df = pq.read_table(
        path,
        columns=["game_slug", "split", "seconds_since_game_start", *X_COLS, *Y_COLS],
    ).to_pandas()
    df = df[df["seconds_since_game_start"] == T_TARGET]
    if exclude_slugs is not None:
        df = df[~df["game_slug"].isin(exclude_slugs)]
    X = df[X_COLS].to_numpy(dtype=np.float64)
    y = df[Y_COLS].to_numpy(dtype=np.float64)
    keep = ~(X == 1.0).all(axis=1)
    Y = np.hstack([y[keep], 1.0 - y[keep]])
    return X[keep], Y, df.loc[keep, "game_slug"].to_numpy()


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


def report(name, X_tr, Y_tr, X_va, Y_va):
    pred = fit_topk(X_tr, Y_tr, KEEP_TOP)
    pred_va = pred(X_va)
    yhat = np.clip(pred_va, 0.0, 1.0)

    yes_idx = list(range(12))
    base_yhat = np.tile(Y_tr.mean(axis=0), (len(Y_va), 1))
    base_mse = ((base_yhat - Y_va) ** 2).mean(axis=0)
    mse = ((yhat - Y_va) ** 2).mean(axis=0)
    ss_tot = ((Y_va - Y_va.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ((yhat - Y_va) ** 2).sum(axis=0) / np.where(ss_tot > 0, ss_tot, 1.0)

    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(f"train n={len(Y_tr)}   val n={len(Y_va)}")
    print(f"Mean Brier (12 YES): base={base_mse[yes_idx].mean():.4f}  "
          f"top3={mse[yes_idx].mean():.4f}  R2={r2[yes_idx].mean():+.4f}")

    ask = X_va
    edge = pred_va - ask
    pnl = Y_va - ask
    valid = (ask >= ASK_LO) & (ask <= ASK_HI)

    type_to_slots = {}
    for slot, mt in TYPE_FOR_SLOT.items():
        for s in (slot, slot + 12):
            type_to_slots.setdefault(mt, []).append(s)

    print(f"\n{'thresh':>7s}  {'agg n/$':>16s}  {'mny n/$':>16s}  "
          f"{'spr n/$':>16s}  {'tot n/$':>16s}  {'btts n/$':>16s}")
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
        print(f"{thr:>7.2f}  " + "  ".join(c.rjust(16) for c in cells))


def main():
    self_collected_slugs = set(
        pq.read_table(paths.SELF_COLLECTED_LABELED, columns=["game_slug"])
        .to_pandas()["game_slug"].unique()
    )

    # Variant 1: original cross-eval (train on all telonex, eval on self_collected)
    X_tx, Y_tx, _ = load(paths.TELONEX_LABELED, exclude_slugs=None)
    X_self_collected, Y_self_collected, _ = load(paths.SELF_COLLECTED_LABELED, exclude_slugs=None)
    report("(A) train: telonex ALL, eval: self_collected ALL  [original — has 96.5% slug leakage]",
           X_tx, Y_tx, X_self_collected, Y_self_collected)

    # Variant 2: clean (train on telonex EXCLUDING self_collected slugs, eval on self_collected)
    X_tx2, Y_tx2, _ = load(paths.TELONEX_LABELED, exclude_slugs=self_collected_slugs)
    report("(B) train: telonex MINUS self_collected overlap, eval: self_collected ALL  [clean OOS by game id]",
           X_tx2, Y_tx2, X_self_collected, Y_self_collected)

    # Variant 3: tiny "purely OOS" — eval on the self_collected games NOT in telonex
    self_collected_not_tx = self_collected_slugs - set(
        pq.read_table(paths.TELONEX_LABELED, columns=["game_slug"])
        .to_pandas()["game_slug"].unique()
    )
    X_self_collected_only, Y_self_collected_only, _ = load(paths.SELF_COLLECTED_LABELED, exclude_slugs=self_collected_slugs - self_collected_not_tx)
    if len(Y_self_collected_only) > 0:
        report(f"(C) train: telonex ALL, eval: self_collected ONLY-NOT-IN-telonex (n={len(Y_self_collected_only)})",
               X_tx, Y_tx, X_self_collected_only, Y_self_collected_only)


if __name__ == "__main__":
    main()
