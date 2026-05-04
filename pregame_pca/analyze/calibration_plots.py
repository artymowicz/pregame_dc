"""Calibration plots for the rank-3 PCR pregame model.

For each fire time t in {-25, -10, -2} min, fits the same clean-OOS model
(train: telonex MINUS self_collected overlap, eval: self_collected) and computes:

  ω̂ = (1/N) Σ (o_i - p_i)             — model bias
  ψ̂ = (1/N) Σ (o_i - q_i)             — realised PnL/share at fill
  SE_theory = √(Σ p_i(1-p_i)) / N      — SE of ω̂ under H0 "model calibrated"

Three views per t:
  1. Unconditional (printed to stdout).
  2. Conditional on (pred - ask) > thr  for thr ∈ {0, 0.01, ..., 0.10}.
  3. Conditional on pred bucket (5-cent buckets over [0, 1)).

Each t produces one PNG: strategies/pregame_pca/plots/calibration/t{T}.png
with two panels (threshold sweep + bucket histogram). Eval restricted to
rows where ask ∈ [0.01, 0.99] (drops placeholder/illiquid quotes).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

from pregame_pca import paths
from pregame_pca.constants import X_COLS, Y_COLS

OUT_DIR = paths.PACKAGE_ROOT / "plots" / "calibration"
T_TARGETS = [-1500.0, -600.0, -120.0]   # -25, -10, -2 min
ASK_LO, ASK_HI = 0.01, 0.99
KEEP_TOP = 3
THRESHOLDS = np.arange(0.00, 0.11, 0.01)   # 0..0.10 in 0.01 steps
BUCKET_EDGES = np.arange(0.0, 1.05, 0.05)  # 5-cent buckets


def _load(path: Path, t_target: float, exclude_slugs=None):
    df = pq.read_table(
        path,
        columns=["game_slug", "seconds_since_game_start", *X_COLS, *Y_COLS],
    ).to_pandas()
    df = df[df["seconds_since_game_start"] == t_target]
    if exclude_slugs is not None:
        df = df[~df["game_slug"].isin(exclude_slugs)]
    X = df[X_COLS].to_numpy(dtype=np.float64)
    y = df[Y_COLS].to_numpy(dtype=np.float64)
    keep = ~(X == 1.0).all(axis=1)
    Y = np.hstack([y[keep], 1.0 - y[keep]])
    return X[keep], Y


def fit_predict(t_target: float):
    self_collected_slugs = set(
        pq.read_table(paths.SELF_COLLECTED_LABELED, columns=["game_slug"])
        .to_pandas()["game_slug"].unique()
    )
    X_tr, Y_tr = _load(paths.TELONEX_LABELED, t_target, exclude_slugs=self_collected_slugs)
    X_va, Y_va = _load(paths.SELF_COLLECTED_LABELED, t_target, exclude_slugs=None)

    n = len(X_tr)
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0, ddof=1)
    sd_safe = np.where(sd > 0, sd, 1.0)
    Z_tr = (X_tr - mu) / sd_safe
    F_tr = np.hstack([Z_tr, np.ones((n, 1))])
    beta = np.linalg.lstsq(F_tr, Y_tr, rcond=None)[0].T
    cov_z = np.cov(Z_tr, rowvar=False, ddof=1)
    eigvals_asc, eigvecs_asc = np.linalg.eigh(cov_z)
    U = eigvecs_asc[:, ::-1]
    for j in range(U.shape[1]):
        k = int(np.argmax(np.abs(U[:, j])))
        if U[k, j] < 0:
            U[:, j] *= -1
    bpc = np.zeros_like(beta[:, :24] @ U)
    bpc[:, :KEEP_TOP] = (beta[:, :24] @ U)[:, :KEEP_TOP]
    beta_K = np.hstack([bpc @ U.T, beta[:, 24:25]])

    Z_va = (X_va - mu) / sd_safe
    F_va = np.hstack([Z_va, np.ones((len(Z_va), 1))])
    pred_va = F_va @ beta_K.T
    return X_va, Y_va, pred_va, n


def _stats(p, q, o):
    """Returns (n, ω̂, ψ̂, SE_theory). p is clipped to [0,1] inside."""
    n = len(p)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    p = np.clip(p, 0.0, 1.0)
    omp = float((o - p).mean())
    omq = float((o - q).mean())
    var_th = float((p * (1 - p)).sum())
    se_th = float(np.sqrt(var_th)) / n if var_th > 0 else 0.0
    return n, omp, omq, se_th


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for T in T_TARGETS:
        print(f"\n{'='*70}\nt = {T:+.0f}s ({T/60:+.1f} min)\n{'='*70}")
        X, Y, P, n_train = fit_predict(T)
        print(f"train n={n_train}  val n={len(X)} games  "
              f"({len(X) * 24:,} (game,slot) cells before ask filter)")

        ask = X
        pred = P
        valid = (ask >= ASK_LO) & (ask <= ASK_HI)
        # Flatten to 1-D arrays restricted to valid cells.
        p_flat = np.clip(pred[valid], 0.0, 1.0)
        q_flat = ask[valid]
        o_flat = Y[valid]
        edge_flat = pred[valid] - ask[valid]   # use unclipped pred for edge

        # -------- Unconditional --------
        n, omp, omq, se = _stats(p_flat, q_flat, o_flat)
        z = omp / se if se > 0 else 0.0
        print(f"\n[unconditional, ask∈[0.01,0.99]]")
        print(f"  N = {n:,}")
        print(f"  ω̂ = ⟨o−p⟩    = {omp:+.4f}")
        print(f"  ψ̂ = ⟨o−q⟩    = {omq:+.4f}")
        print(f"  SE_theory     = {se:.4f}    Z = {z:+.2f}")

        # -------- Threshold sweep --------
        thr_rows = []
        for thr in THRESHOLDS:
            mask = edge_flat > thr
            n_, omp_, omq_, se_ = _stats(p_flat[mask], q_flat[mask], o_flat[mask])
            thr_rows.append((float(thr), n_, omp_, omq_, se_))
        print(f"\n[edge > thr]   thr   N      ω̂        ψ̂        SE_theory  Z")
        for thr, n_, omp_, omq_, se_ in thr_rows:
            z_ = omp_ / se_ if se_ > 0 else 0.0
            print(f"               {thr:.2f}  {n_:>5d}  {omp_:+.4f}  "
                  f"{omq_:+.4f}  {se_:.4f}    {z_:+.2f}")

        # -------- Pred-bucket --------
        bucket_rows = []
        for i in range(len(BUCKET_EDGES) - 1):
            lo, hi = BUCKET_EDGES[i], BUCKET_EDGES[i + 1]
            mask = (p_flat >= lo) & (p_flat < hi if hi < 1.0 else p_flat <= hi)
            n_, omp_, omq_, se_ = _stats(p_flat[mask], q_flat[mask], o_flat[mask])
            center = (lo + hi) / 2
            bucket_rows.append((center, n_, omp_, omq_, se_))
        print(f"\n[pred bucket]  center  N      ω̂        ψ̂        SE_theory")
        for c, n_, omp_, omq_, se_ in bucket_rows:
            print(f"               {c:.3f}   {n_:>5d}  {omp_:+.4f}  "
                  f"{omq_:+.4f}  {se_:.4f}")

        # -------- Plot --------
        fig, (ax_t, ax_b) = plt.subplots(1, 2, figsize=(14, 5))

        # Threshold panel
        thrs = np.array([r[0] for r in thr_rows])
        ns = np.array([r[1] for r in thr_rows])
        omps = np.array([r[2] for r in thr_rows])
        omqs = np.array([r[3] for r in thr_rows])
        ses = np.array([r[4] for r in thr_rows])
        ax_t.axhline(0.0, color="black", lw=0.5)
        # ω̂ and ψ̂ have the same theoretical SE under H0 (Var(o)=p(1-p)),
        # so reuse `ses` for both.
        ax_t.errorbar(thrs, omps, yerr=ses, fmt="o", ls="none", capsize=3,
                      color="#1f77b4", label="ω̂ = ⟨o−p⟩  (±1 SE_theory)")
        ax_t.errorbar(thrs, omqs, yerr=ses, fmt="s", ls="none", capsize=3,
                      color="#d62728", label="ψ̂ = ⟨o−q⟩  (±1 SE_theory)")
        ax_t.set_xlabel("edge threshold (pred − ask)")
        ax_t.set_ylabel("¢/share")
        ax_t.set_title(f"calibration vs edge threshold  (t={T/60:+.1f} min)")
        ax_t.grid(alpha=0.3)
        ax_t.legend(loc="best", fontsize=9)
        for x, y, n_ in zip(thrs, omps, ns):
            ax_t.annotate(f"{n_}", (x, y), textcoords="offset points",
                          xytext=(0, 8), fontsize=7, ha="center", color="#1f77b4")

        # Bucket panel
        cs = np.array([r[0] for r in bucket_rows])
        ns_b = np.array([r[1] for r in bucket_rows])
        omps_b = np.array([r[2] for r in bucket_rows])
        omqs_b = np.array([r[3] for r in bucket_rows])
        ses_b = np.array([r[4] for r in bucket_rows])
        valid_b = ns_b > 0
        ax_b.axhline(0.0, color="black", lw=0.5)
        ax_b.errorbar(cs[valid_b], omps_b[valid_b], yerr=ses_b[valid_b],
                      fmt="o", ls="none", capsize=3,
                      color="#1f77b4", label="ω̂ = ⟨o−p⟩  (±1 SE_theory)")
        ax_b.errorbar(cs[valid_b], omqs_b[valid_b], yerr=ses_b[valid_b],
                      fmt="s", ls="none", capsize=3,
                      color="#d62728", label="ψ̂ = ⟨o−q⟩  (±1 SE_theory)")
        ax_b.set_xlabel("predicted prob (5-cent buckets, center)")
        ax_b.set_ylabel("¢/share")
        ax_b.set_title(f"calibration vs pred bucket  (t={T/60:+.1f} min)")
        ax_b.grid(alpha=0.3)
        ax_b.legend(loc="best", fontsize=9)
        for x, y, n_ in zip(cs[valid_b], omps_b[valid_b], ns_b[valid_b]):
            ax_b.annotate(f"{n_}", (x, y), textcoords="offset points",
                          xytext=(0, 8), fontsize=7, ha="center", color="#1f77b4")

        fig.suptitle(
            f"rank-3 PCR calibration  |  t={T/60:+.1f} min  "
            f"|  train n={n_train}  val cells={len(p_flat):,} "
            f"(after ask∈[{ASK_LO},{ASK_HI}] filter)",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = OUT_DIR / f"t{int(T)}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
