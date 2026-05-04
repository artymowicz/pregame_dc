"""Clean OOS rank-3 PCR cross-eval at multiple t offsets.

Train: telonex games not present in self_collected.
Eval:  self_collected.

For each t in {-1500, -600, -120}s (i.e. -25 / -10 / -2 min), reports:
  - Brier R² over the 12 YES marginals
  - Threshold-sweep PnL/trade by market type (moneyline/spread/totals/btts)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pregame_pca import paths
from pregame_pca.constants import X_COLS, Y_COLS, TYPE_FOR_SLOT

ASK_LO, ASK_HI = 0.01, 0.99
THRESHOLDS = [0.00, 0.02, 0.05, 0.10, 0.15]
KEEP_TOP = 3

T_TARGETS = [-1500.0, -600.0, -120.0]   # seconds: -25, -10, -2 min


def load(path: Path, t_target: float, exclude_slugs=None):
    """Returns (X, Y, slugs) at the given t-per_game_data.

    X: (n, 24) ask matrix. Y: (n, 24) outcome matrix (12 YES then 12 NO).
    Rows where every ask is 1.0 (placeholder pre-listing rows) are dropped.
    """
    df = pq.read_table(
        path,
        columns=["game_slug", "split", "seconds_since_game_start", *X_COLS, *Y_COLS],
    ).to_pandas()
    df = df[df["seconds_since_game_start"] == t_target]
    if exclude_slugs is not None:
        df = df[~df["game_slug"].isin(exclude_slugs)]
    X = df[X_COLS].to_numpy(dtype=np.float64)
    y = df[Y_COLS].to_numpy(dtype=np.float64)
    keep = ~(X == 1.0).all(axis=1)
    X = X[keep]
    y = y[keep]
    Y = np.hstack([y, 1.0 - y])
    slugs = df.loc[keep, "game_slug"].to_numpy()
    return X, Y, slugs


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


def _calib_stats(p, q, o):
    n = len(p)
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    p = np.clip(p, 0.0, 1.0)
    omp = o - p
    var_th = float((p * (1 - p)).sum())
    se_th = (var_th ** 0.5) / n if var_th > 0 else 0.0
    var_sm = float((omp * omp).sum())
    se_sm = (var_sm ** 0.5) / n if var_sm > 0 else 0.0
    return n, float(omp.mean()), se_th, se_sm, float((o - q).mean())


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

    print(f"\n{'='*70}\n{name}\n{'='*70}")
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

    # ---- Calibration stats ----
    print(f"\nCalibration  (p_clipped vs o, q=ask paid;  ω̂=⟨o−p⟩, ψ̂=⟨o−q⟩)")
    print(f"  fmt: thr   subset            N    ω̂        SE_theory  Z         ψ̂")
    for thr in [0.05, 0.10]:
        fire = valid & (edge > thr)
        rows = [("ALL    ", fire)]
        for mt in ("moneyline", "spread", "totals", "btts"):
            slots = type_to_slots[mt]
            mask = np.zeros_like(fire)
            mask[:, slots] = fire[:, slots]
            rows.append((f"{mt:<7s}", mask))

        for label, mask in rows:
            p = yhat[mask]
            q = ask[mask]
            o = Y_va[mask]
            n, omp, se_th, se_sm, omq = _calib_stats(p, q, o)
            if n == 0:
                print(f"  {thr:>4.2f}  {label}  n=0")
                continue
            z = omp / se_th if se_th > 0 else 0.0
            verdict = "PASS" if abs(z) < 2 else "FAIL"
            print(f"  {thr:>4.2f}  {label}  N={n:>4d}  ω̂={omp:+.4f}  "
                  f"SE_th={se_th:.4f}  Z={z:+.2f} ({verdict})  ψ̂={omq:+.4f}")


def main():
    self_collected_slugs = set(
        pq.read_table(paths.SELF_COLLECTED_LABELED, columns=["game_slug"])
        .to_pandas()["game_slug"].unique()
    )
    for T in T_TARGETS:
        X_tx, Y_tx, _ = load(paths.TELONEX_LABELED, T, exclude_slugs=self_collected_slugs)
        X_self_collected, Y_self_collected, _ = load(paths.SELF_COLLECTED_LABELED, T, exclude_slugs=None)
        report(
            f"clean OOS @ t={int(T)}s ({T/60:+.1f} min)  "
            f"train: telonex MINUS self_collected overlap, eval: self_collected",
            X_tx, Y_tx, X_self_collected, Y_self_collected,
        )


if __name__ == "__main__":
    main()
