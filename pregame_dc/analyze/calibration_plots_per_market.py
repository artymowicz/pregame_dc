"""Per-market-type calibration plots for the rank-3 PCR pregame model.

Same logic as calibration_plots.py but with one figure per market type
(moneyline / spread / totals / btts) showing all three t slices in rows
and {threshold sweep, pred bucket} in columns.

Slots per market type (canonical 0..23 indexing from convex_arb):
  moneyline: 0,1,2  (YES) + 12,13,14 (NO)
  spread:    3,4,5,6  (YES) + 15,16,17,18 (NO)
  totals:    7,8,9,10  (YES) + 19,20,21,22 (NO)
  btts:      11 (YES) + 23 (NO)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

from pregame_dc import paths
from pregame_dc.constants import X_COLS, Y_COLS

OUT_DIR = paths.PACKAGE_ROOT / "plots" / "calibration_per_market"
T_TARGETS = [-1500.0, -600.0, -120.0]
ASK_LO, ASK_HI = 0.01, 0.99
KEEP_TOP = 3
THRESHOLDS = np.arange(0.00, 0.11, 0.01)
BUCKET_EDGES = np.arange(0.0, 1.05, 0.05)

YES_SLOTS_BY_TYPE = {
    "moneyline": [0, 1, 2],
    "spread":    [3, 4, 5, 6],
    "totals":    [7, 8, 9, 10],
    "btts":      [11],
}
SLOTS_BY_TYPE = {
    mt: ys + [s + 12 for s in ys] for mt, ys in YES_SLOTS_BY_TYPE.items()
}


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
    n = len(p)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    p = np.clip(p, 0.0, 1.0)
    omp = float((o - p).mean())
    omq = float((o - q).mean())
    var_th = float((p * (1 - p)).sum())
    se_th = float(np.sqrt(var_th)) / n if var_th > 0 else 0.0
    return n, omp, omq, se_th


def _select_market_cells(ask, pred, Y, slots):
    """Restrict (game, slot) cells to slots in `slots` AND ask in [ASK_LO, ASK_HI].
    Returns flat (p_clipped, q, o, edge_unclipped) arrays."""
    sub_mask = np.zeros_like(ask, dtype=bool)
    sub_mask[:, slots] = True
    valid = (ask >= ASK_LO) & (ask <= ASK_HI) & sub_mask
    p = np.clip(pred[valid], 0.0, 1.0)
    q = ask[valid]
    o = Y[valid]
    edge = pred[valid] - ask[valid]
    return p, q, o, edge


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Compute model + cell arrays for each T once
    per_t = {}
    for T in T_TARGETS:
        print(f"fitting + predicting at t={T:+.0f}s ...")
        X, Y, P, n_train = fit_predict(T)
        per_t[T] = dict(X=X, Y=Y, P=P, n_train=n_train)

    for mt, slots in SLOTS_BY_TYPE.items():
        fig, axes = plt.subplots(len(T_TARGETS), 2, figsize=(14, 4.2 * len(T_TARGETS)))
        for row, T in enumerate(T_TARGETS):
            d = per_t[T]
            p_flat, q_flat, o_flat, edge_flat = _select_market_cells(
                d["X"], d["P"], d["Y"], slots
            )

            # Unconditional
            n_u, omp_u, omq_u, se_u = _stats(p_flat, q_flat, o_flat)
            print(f"\n[{mt:<9s} t={T:+.0f}s]  uncond N={n_u:>5d}  "
                  f"ω̂={omp_u:+.4f}  ψ̂={omq_u:+.4f}  SE_th={se_u:.4f}")

            # Threshold sweep
            thr_rows = []
            for thr in THRESHOLDS:
                mask = edge_flat > thr
                n_, omp_, omq_, se_ = _stats(p_flat[mask], q_flat[mask], o_flat[mask])
                thr_rows.append((float(thr), n_, omp_, omq_, se_))

            # Bucket
            bucket_rows = []
            for i in range(len(BUCKET_EDGES) - 1):
                lo, hi = BUCKET_EDGES[i], BUCKET_EDGES[i + 1]
                if hi < 1.0:
                    mask = (p_flat >= lo) & (p_flat < hi)
                else:
                    mask = (p_flat >= lo) & (p_flat <= hi)
                n_, omp_, omq_, se_ = _stats(p_flat[mask], q_flat[mask], o_flat[mask])
                bucket_rows.append(((lo + hi) / 2, n_, omp_, omq_, se_))

            ax_t, ax_b = axes[row]

            # Threshold panel
            thrs = np.array([r[0] for r in thr_rows])
            ns = np.array([r[1] for r in thr_rows])
            omps = np.array([r[2] for r in thr_rows])
            omqs = np.array([r[3] for r in thr_rows])
            ses = np.array([r[4] for r in thr_rows])
            ax_t.axhline(0.0, color="black", lw=0.5)
            # ω̂ and ψ̂ have the same theoretical SE under H0 (Var(o)=p(1-p)).
            ax_t.errorbar(thrs, omps, yerr=ses, fmt="o", ls="none", capsize=3,
                          color="#1f77b4", label="ω̂ = ⟨o−p⟩  (±1 SE_theory)")
            ax_t.errorbar(thrs, omqs, yerr=ses, fmt="s", ls="none", capsize=3,
                          color="#d62728", label="ψ̂ = ⟨o−q⟩  (±1 SE_theory)")
            ax_t.set_xlabel("edge threshold (pred − ask)")
            ax_t.set_ylabel("¢/share")
            ax_t.set_title(f"{mt}  |  vs threshold  (t={T/60:+.1f} min, "
                           f"uncond N={n_u})")
            ax_t.grid(alpha=0.3)
            ax_t.legend(loc="best", fontsize=8)
            for x, y, n_ in zip(thrs, omps, ns):
                if n_ > 0:
                    ax_t.annotate(f"{n_}", (x, y), textcoords="offset points",
                                  xytext=(0, 8), fontsize=7, ha="center",
                                  color="#1f77b4")

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
            ax_b.set_title(f"{mt}  |  vs pred bucket  (t={T/60:+.1f} min)")
            ax_b.grid(alpha=0.3)
            ax_b.legend(loc="best", fontsize=8)
            for x, y, n_ in zip(cs[valid_b], omps_b[valid_b], ns_b[valid_b]):
                ax_b.annotate(f"{n_}", (x, y), textcoords="offset points",
                              xytext=(0, 8), fontsize=7, ha="center",
                              color="#1f77b4")

        fig.suptitle(
            f"rank-3 PCR calibration — market type: {mt.upper()}  "
            f"(slots {SLOTS_BY_TYPE[mt]})",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = OUT_DIR / f"{mt}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
