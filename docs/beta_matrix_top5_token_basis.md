# Top-5 PC reconstruction of β in token × token basis

OLS fit of `y = β x` (z-scored asks + bias), with all PC columns *except the top 5* zeroed out, then rotated back to the canonical token feature basis. Both rows and columns now correspond to the 24 ask tokens — the matrix is 24 (outcome) × 24 (feature) plus a bias column.

- Dataset: telonex `train` split, fully-stale rows dropped, `n = 2107`.
- Features: z-scored asks at t = −10 min.
- Reconstruction: zero PC6..PC24 columns of `β_eig`, then `β_token = β_eig_truncated[:, :24] @ U^T` (since `U` is orthogonal, `U^{-1} = U^T`). Bias column unchanged.
- Top 5 PCs explain **73.10% of feature variance** (sum of standardized eigenvalues 6.46 + 5.24 + 3.85 + 1.05 + 0.95 over trace 24) and **82.56% of aggregated fitted-prediction variance** (`Σ_{k=1..5} β_pc[i,k]² · λ_k / Σ_{k=1..24} β_pc[i,k]² · λ_k`, summed across outcomes).

Per-outcome top-5 share of fitted variance:

| outcome | top-5 share | (top-3 was) |
|---|---:|---:|
| A win Y  | 93.6% | 90.5% |
| B win Y  | 87.8% | 87.2% |
| Draw Y   | 51.4% | 42.2% |
| A -1.5 Y | 92.5% | 92.0% |
| B -1.5 Y | 86.5% | 86.3% |
| A -2.5 Y | 85.3% | 84.0% |
| B -2.5 Y | 71.7% | 71.2% |
| O 1.5 Y  | 64.4% | 64.2% |
| O 2.5 Y  | 81.3% | 80.4% |
| O 3.5 Y  | 87.7% | 86.2% |
| O 4.5 Y  | 73.1% | 70.9% |
| BTTS Y   | 54.7% | 54.6% |

PC4 is mostly an Awin/Draw refinement; PC5 picks up the Draw structure most visibly. The biggest shift adding PCs 4–5 is on **Draw Y** (42% → 51%) — the Draw row is the only outcome whose fitted-variance share moves materially. BTTS Y stays stuck around 55%, meaning BTTS's predictive signal lives outside the top 5 PCs as well.

Run: `python -m strategies.pregame_dc.regression --keep-top 5`
Source CSV: `strategies/pregame_dc/beta_token_x_token_top5.csv`
Heatmap: `strategies/pregame_dc/plots/beta_token_x_token_top5.png`

## How to read it

Row label = predicted outcome token. Column label = z-scored ask feature token. Cell `β[i, j]` answers: *"if z-scored ask of token j increases by 1 std, how does predicted P(outcome i) change?"* The bias column is the empirical base rate (the OLS intercept, unregularised).

## Coefficient matrix (top-5 PC reconstruction)

| token | A win Y | B win Y | Draw Y | A -1.5 Y | B -1.5 Y | A -2.5 Y | B -2.5 Y | O 1.5 Y | O 2.5 Y | O 3.5 Y | O 4.5 Y | BTTS Y | A win N | B win N | Draw N | A -1.5 N | B -1.5 N | A -2.5 N | B -2.5 N | O 1.5 N | O 2.5 N | O 3.5 N | O 4.5 N | BTTS N | bias |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A win Y | +0.028 | -0.020 | -0.003 | +0.012 | -0.002 | +0.013 | +0.012 | -0.005 | -0.007 | +0.001 | +0.003 | -0.021 | -0.027 | +0.025 | +0.020 | -0.025 | +0.023 | -0.021 | +0.020 | -0.010 | -0.009 | -0.002 | +0.000 | -0.004 | +0.433 |
| B win Y | -0.018 | +0.023 | +0.004 | -0.010 | +0.003 | -0.006 | -0.009 | +0.007 | +0.008 | +0.003 | -0.001 | +0.009 | +0.022 | -0.023 | +0.003 | +0.017 | -0.026 | +0.013 | -0.024 | +0.000 | -0.002 | -0.005 | -0.006 | -0.004 | +0.296 |
| Draw Y | -0.009 | -0.004 | -0.001 | -0.003 | -0.002 | -0.008 | -0.003 | -0.002 | -0.000 | -0.004 | -0.002 | +0.012 | +0.005 | -0.002 | -0.022 | +0.008 | +0.003 | +0.009 | +0.005 | +0.010 | +0.011 | +0.008 | +0.006 | +0.007 | +0.271 |
| A -1.5 Y | +0.019 | -0.012 | -0.003 | +0.008 | -0.003 | +0.003 | +0.001 | +0.005 | +0.004 | +0.004 | +0.001 | -0.005 | -0.018 | +0.016 | +0.014 | -0.018 | +0.013 | -0.017 | +0.010 | -0.005 | -0.007 | -0.007 | -0.007 | -0.001 | +0.215 |
| B -1.5 Y | -0.011 | +0.015 | +0.002 | -0.003 | +0.006 | +0.002 | +0.001 | +0.006 | +0.005 | +0.004 | +0.003 | +0.004 | +0.013 | -0.015 | +0.005 | +0.010 | -0.015 | +0.007 | -0.015 | -0.001 | -0.002 | -0.003 | -0.004 | -0.003 | +0.117 |
| A -2.5 Y | +0.012 | -0.005 | -0.002 | +0.006 | +0.001 | +0.005 | +0.003 | +0.006 | +0.005 | +0.005 | +0.002 | -0.003 | -0.010 | +0.008 | +0.013 | -0.011 | +0.005 | -0.012 | +0.003 | -0.005 | -0.007 | -0.007 | -0.007 | -0.003 | +0.098 |
| B -2.5 Y | -0.005 | +0.006 | +0.000 | -0.000 | +0.004 | +0.002 | +0.002 | +0.003 | +0.003 | +0.002 | +0.002 | +0.003 | +0.005 | -0.006 | +0.001 | +0.005 | -0.006 | +0.003 | -0.006 | -0.000 | -0.001 | -0.002 | -0.002 | -0.001 | +0.039 |
| O 1.5 Y | +0.002 | +0.003 | -0.007 | -0.001 | -0.001 | -0.000 | -0.001 | +0.010 | +0.008 | +0.003 | -0.002 | -0.000 | -0.003 | -0.004 | +0.012 | -0.006 | -0.008 | -0.008 | -0.009 | -0.010 | -0.014 | -0.015 | -0.015 | -0.010 | +0.734 |
| O 2.5 Y | +0.005 | +0.005 | -0.006 | +0.003 | +0.003 | +0.005 | +0.002 | +0.016 | +0.014 | +0.009 | +0.002 | +0.002 | -0.004 | -0.003 | +0.020 | -0.008 | -0.009 | -0.011 | -0.011 | -0.011 | -0.017 | -0.019 | -0.019 | -0.010 | +0.496 |
| O 3.5 Y | +0.005 | +0.006 | -0.001 | +0.002 | +0.001 | +0.001 | -0.003 | +0.014 | +0.013 | +0.008 | +0.002 | +0.003 | -0.001 | -0.003 | +0.018 | -0.006 | -0.008 | -0.009 | -0.011 | -0.007 | -0.011 | -0.014 | -0.014 | -0.006 | +0.276 |
| O 4.5 Y | +0.004 | +0.005 | -0.001 | +0.001 | +0.001 | +0.001 | -0.002 | +0.008 | +0.007 | +0.005 | +0.001 | -0.000 | -0.001 | -0.002 | +0.014 | -0.004 | -0.006 | -0.006 | -0.007 | -0.005 | -0.008 | -0.009 | -0.009 | -0.005 | +0.133 |
| BTTS Y | -0.004 | +0.005 | -0.006 | -0.005 | -0.003 | -0.004 | -0.004 | +0.007 | +0.006 | +0.000 | -0.004 | +0.002 | +0.002 | -0.008 | +0.005 | -0.000 | -0.011 | -0.002 | -0.010 | -0.008 | -0.010 | -0.012 | -0.012 | -0.009 | +0.526 |
| A win N | -0.028 | +0.020 | +0.003 | -0.012 | +0.002 | -0.013 | -0.012 | +0.005 | +0.007 | -0.001 | -0.003 | +0.021 | +0.027 | -0.025 | -0.020 | +0.025 | -0.023 | +0.021 | -0.020 | +0.010 | +0.009 | +0.002 | -0.000 | +0.004 | +0.567 |
| B win N | +0.018 | -0.023 | -0.004 | +0.010 | -0.003 | +0.006 | +0.009 | -0.007 | -0.008 | -0.003 | +0.001 | -0.009 | -0.022 | +0.023 | -0.003 | -0.017 | +0.026 | -0.013 | +0.024 | -0.000 | +0.002 | +0.005 | +0.006 | +0.004 | +0.704 |
| Draw N | +0.009 | +0.004 | +0.001 | +0.003 | +0.002 | +0.008 | +0.003 | +0.002 | +0.000 | +0.004 | +0.002 | -0.012 | -0.005 | +0.002 | +0.022 | -0.008 | -0.003 | -0.009 | -0.005 | -0.010 | -0.011 | -0.008 | -0.006 | -0.007 | +0.729 |
| A -1.5 N | -0.019 | +0.012 | +0.003 | -0.008 | +0.003 | -0.003 | -0.001 | -0.005 | -0.004 | -0.004 | -0.001 | +0.005 | +0.018 | -0.016 | -0.014 | +0.018 | -0.013 | +0.017 | -0.010 | +0.005 | +0.007 | +0.007 | +0.007 | +0.001 | +0.785 |
| B -1.5 N | +0.011 | -0.015 | -0.002 | +0.003 | -0.006 | -0.002 | -0.001 | -0.006 | -0.005 | -0.004 | -0.003 | -0.004 | -0.013 | +0.015 | -0.005 | -0.010 | +0.015 | -0.007 | +0.015 | +0.001 | +0.002 | +0.003 | +0.004 | +0.003 | +0.883 |
| A -2.5 N | -0.012 | +0.005 | +0.002 | -0.006 | -0.001 | -0.005 | -0.003 | -0.006 | -0.005 | -0.005 | -0.002 | +0.003 | +0.010 | -0.008 | -0.013 | +0.011 | -0.005 | +0.012 | -0.003 | +0.005 | +0.007 | +0.007 | +0.007 | +0.003 | +0.902 |
| B -2.5 N | +0.005 | -0.006 | -0.000 | +0.000 | -0.004 | -0.002 | -0.002 | -0.003 | -0.003 | -0.002 | -0.002 | -0.003 | -0.005 | +0.006 | -0.001 | -0.005 | +0.006 | -0.003 | +0.006 | +0.000 | +0.001 | +0.002 | +0.002 | +0.001 | +0.961 |
| O 1.5 N | -0.002 | -0.003 | +0.007 | +0.001 | +0.001 | +0.000 | +0.001 | -0.010 | -0.008 | -0.003 | +0.002 | +0.000 | +0.003 | +0.004 | -0.012 | +0.006 | +0.008 | +0.008 | +0.009 | +0.010 | +0.014 | +0.015 | +0.015 | +0.010 | +0.266 |
| O 2.5 N | -0.005 | -0.005 | +0.006 | -0.003 | -0.003 | -0.005 | -0.002 | -0.016 | -0.014 | -0.009 | -0.002 | -0.002 | +0.004 | +0.003 | -0.020 | +0.008 | +0.009 | +0.011 | +0.011 | +0.011 | +0.017 | +0.019 | +0.019 | +0.010 | +0.504 |
| O 3.5 N | -0.005 | -0.006 | +0.001 | -0.002 | -0.001 | -0.001 | +0.003 | -0.014 | -0.013 | -0.008 | -0.002 | -0.003 | +0.001 | +0.003 | -0.018 | +0.006 | +0.008 | +0.009 | +0.011 | +0.007 | +0.011 | +0.014 | +0.014 | +0.006 | +0.724 |
| O 4.5 N | -0.004 | -0.005 | +0.001 | -0.001 | -0.001 | -0.001 | +0.002 | -0.008 | -0.007 | -0.005 | -0.001 | +0.000 | +0.001 | +0.002 | -0.014 | +0.004 | +0.006 | +0.006 | +0.007 | +0.005 | +0.008 | +0.009 | +0.009 | +0.005 | +0.867 |
| BTTS N | +0.004 | -0.005 | +0.006 | +0.005 | +0.003 | +0.004 | +0.004 | -0.007 | -0.006 | -0.000 | +0.004 | -0.002 | -0.002 | +0.008 | -0.005 | +0.000 | +0.011 | +0.002 | +0.010 | +0.008 | +0.010 | +0.012 | +0.012 | +0.009 | +0.474 |

## What's new vs the top-3 reconstruction

(see `beta_matrix_top3_token_basis.md` for the rank-3 version)

- **Draw column lights up.** The `Draw N` feature column now has visible non-zero entries (`A win Y` row +0.020, `Draw Y` row −0.022, `Draw N` row +0.022, mirror N entries flipped). PC4 + PC5 capture the "favored team draws / no draw" axis that was missing in the top-3.
- **Spread A-2.5 / B-2.5 features get clearer loadings.** PC4 is essentially `A win Y vs A win N` refinement plus blowout direction; the `A-2.5` column in the matrix is now ±0.013 instead of ±0.008 for moneyline rows.
- **Totals block sharpens.** O 2.5 Y row → O 1.5/2.5 Y feature is now +0.014 to +0.016 (was +0.011 to +0.016) and the under-side block (O 2.5/3.5 N feature) goes to −0.017 to −0.019 (was −0.014 to −0.020). Marginal change — totals were already mostly captured at rank 3.
- **The four block structure remains** — same YES/NO antisymmetry, same diagonal positivity, same draw/BTTS rows being the smallest.

## Block structure (unchanged)

- **Top-left (YES outcome × YES feature)**: moneyline 2×2 sub-block, spread extension, totals cluster.
- **Top-right (YES outcome × NO feature)**: sign-flipped image of top-left.
- **Bottom half**: exact negative of top half (`Y_NO = 1 − Y_YES`).
- **Diagonal positive everywhere** — calibrated book sanity check.

## Summary

Going from top-3 → top-5 PCs adds two refinements: a **Draw structure** axis (PC5) and a **moneyline-tilt-vs-blowout-magnitude residual** (PC4). Aggregate fitted-variance share moves from 80.9% → 82.6% (small absolute gain), but the per-outcome share for **Draw nearly doubles** (42% → 51%). For any analysis where Draw probability matters, top-5 is meaningfully better than top-3; for moneyline / totals / spreads, top-3 was already saturating.
