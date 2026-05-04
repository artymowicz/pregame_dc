"""Compare rank-3 PCR fits across t = -25, -10, -2 min.

For each pair of time slices, reports:
  - top-3 eigenvalue ratios (how much PC variance shifts with t)
  - cosine similarity between matching PC eigenvectors (sign-aligned)
  - element-wise beta_K differences (max abs, mean abs, Frobenius)
  - per-output-token cosine similarity between beta rows

Also prints the full 3-PC scores matrix at each t (24 outputs × 3 PCs)
so you can see whether the regression directions move.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pregame_pca import paths
from pregame_pca.constants import X_COLS, Y_COLS

T_TARGETS = [-1500.0, -600.0, -120.0]
KEEP_TOP = 3

SLOT_LABELS = [
    "A_win", "B_win", "Draw", "A-1.5", "B-1.5", "A-2.5", "B-2.5",
    "Over1.5", "Over2.5", "Over3.5", "Over4.5", "BTTS",
]


def fit_at(t_target: float):
    self_collected_slugs = set(
        pq.read_table(paths.SELF_COLLECTED_LABELED, columns=["game_slug"])
        .to_pandas()["game_slug"].unique()
    )
    df = pq.read_table(
        paths.TELONEX_LABELED,
        columns=["game_slug", "seconds_since_game_start", *X_COLS, *Y_COLS],
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
    beta = np.linalg.lstsq(F, Y_tr, rcond=None)[0].T  # (24, 25)
    cov_z = np.cov(Z, rowvar=False, ddof=1)
    eigvals_asc, eigvecs_asc = np.linalg.eigh(cov_z)
    eigvals = eigvals_asc[::-1]
    U = eigvecs_asc[:, ::-1]
    for j in range(U.shape[1]):
        k = int(np.argmax(np.abs(U[:, j])))
        if U[k, j] < 0:
            U[:, j] *= -1
    beta_pc = beta[:, :24] @ U          # (24 outputs × 24 PCs)
    bpc_K = np.zeros_like(beta_pc)
    bpc_K[:, :KEEP_TOP] = beta_pc[:, :KEEP_TOP]
    beta_K = np.hstack([bpc_K @ U.T, beta[:, 24:25]])  # (24, 25)

    return dict(
        t=t_target, n=n, mu=mu, sd_safe=sd_safe, beta=beta,
        eigvals=eigvals, U=U, beta_pc=beta_pc, beta_K=beta_K,
    )


def _sign_align(u_a: np.ndarray, u_b: np.ndarray) -> np.ndarray:
    """Flip signs of u_b columns so that diag(U_a^T U_b) is non-negative."""
    s = np.sign((u_a * u_b).sum(axis=0))
    s[s == 0] = 1.0
    return u_b * s


def main():
    fits = [fit_at(T) for T in T_TARGETS]

    print("\n=== Top-5 eigenvalues of standardized cov(Z) ===")
    print(f"{'t':>9s}  " + "  ".join(f"λ{i+1:>1d}" for i in range(5))
          + "    Σλ_top3 / trace")
    for f in fits:
        ev = f['eigvals']
        cum = ev[:KEEP_TOP].sum() / ev.sum()
        row = "  ".join(f"{v:6.3f}" for v in ev[:5])
        print(f"  t={int(f['t']):+5d}s  {row}    {cum*100:.1f}%")

    print("\n=== Cosine similarity of top-3 PCs across time ===")
    print("(after sign alignment; values close to 1.0 mean PCs are stable)")
    for i in range(len(fits)):
        for j in range(i + 1, len(fits)):
            U_i, U_j = fits[i]['U'][:, :KEEP_TOP], fits[j]['U'][:, :KEEP_TOP]
            U_j_aligned = _sign_align(U_i, U_j)
            cos = (U_i * U_j_aligned).sum(axis=0)
            ti, tj = int(fits[i]['t']), int(fits[j]['t'])
            print(f"  cos(PC_t={ti}, PC_t={tj})  =  "
                  + "  ".join(f"PC{k+1}: {cos[k]:+.4f}" for k in range(KEEP_TOP)))

    print("\n=== mu and sd_safe across time (per-feature) ===")
    print("Frobenius / max-abs differences:")
    for i in range(len(fits)):
        for j in range(i + 1, len(fits)):
            d_mu = fits[i]['mu'] - fits[j]['mu']
            d_sd = fits[i]['sd_safe'] - fits[j]['sd_safe']
            ti, tj = int(fits[i]['t']), int(fits[j]['t'])
            print(f"  t={ti}s vs t={tj}s   "
                  f"|Δmu|_max={np.abs(d_mu).max():.4f}  "
                  f"|Δmu|_mean={np.abs(d_mu).mean():.4f}    "
                  f"|Δsd|_max={np.abs(d_sd).max():.4f}  "
                  f"|Δsd|_mean={np.abs(d_sd).mean():.4f}")

    print("\n=== beta_K (24 × 25) coefficient comparisons ===")
    for i in range(len(fits)):
        for j in range(i + 1, len(fits)):
            B_i, B_j = fits[i]['beta_K'], fits[j]['beta_K']
            d = B_i - B_j
            ti, tj = int(fits[i]['t']), int(fits[j]['t'])
            print(f"  t={ti}s vs t={tj}s   "
                  f"max|Δβ|={np.abs(d).max():.4f}  "
                  f"mean|Δβ|={np.abs(d).mean():.4f}  "
                  f"||Δβ||_F={np.linalg.norm(d):.4f}    "
                  f"||B_i||_F={np.linalg.norm(B_i):.4f}  "
                  f"||B_j||_F={np.linalg.norm(B_j):.4f}    "
                  f"rel={np.linalg.norm(d)/np.linalg.norm(B_i):.3f}")

    print("\n=== Per-output-token cosine similarity of beta_K rows ===")
    print("Row k of beta_K is the linear coefficients (over z-scored asks + bias)")
    print("predicting outcome k. cos≈1 means the predictor is the same across t.\n")
    print(f"  slot  label      cos(t=-25, t=-10)  cos(t=-25, t=-2)  cos(t=-10, t=-2)")
    cos_a = []
    cos_b = []
    cos_c = []
    for k in range(24):
        rows = [f['beta_K'][k] for f in fits]
        def _c(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return (a @ b) / (na * nb) if na > 0 and nb > 0 else 0.0
        c_ab = _c(rows[0], rows[1])
        c_ac = _c(rows[0], rows[2])
        c_bc = _c(rows[1], rows[2])
        cos_a.append(c_ab); cos_b.append(c_ac); cos_c.append(c_bc)
        side = "Y" if k < 12 else "N"
        label = SLOT_LABELS[k % 12] + "/" + side
        print(f"  {k:>4d}  {label:<10s}  {c_ab:+.4f}            {c_ac:+.4f}            {c_bc:+.4f}")
    print(f"  ----  mean       {np.mean(cos_a):+.4f}            "
          f"{np.mean(cos_b):+.4f}            {np.mean(cos_c):+.4f}")

    print("\n=== Top-3 PC regression scores (beta_pc[:, :3]) per output ===")
    print("Each cell is how much output token k loads on PC j (z-score basis).\n")
    for f in fits:
        print(f"\n--- t = {int(f['t']):+5d}s ---")
        print(f"  {'slot':>4s}  {'label':<10s}  {'PC1':>8s}  {'PC2':>8s}  {'PC3':>8s}")
        for k in range(24):
            side = "Y" if k < 12 else "N"
            label = SLOT_LABELS[k % 12] + "/" + side
            row = f['beta_pc'][k, :KEEP_TOP]
            print(f"  {k:>4d}  {label:<10s}  "
                  + "  ".join(f"{v:+8.4f}" for v in row))


if __name__ == "__main__":
    main()
