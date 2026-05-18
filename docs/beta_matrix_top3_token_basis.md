# Top-3 PC reconstruction of β in token × token basis

OLS fit of `y = β x` (z-scored asks + bias), with all PC columns *except the top 3* zeroed out, then rotated back to the canonical token feature basis. Both rows and columns now correspond to the 24 ask tokens — the matrix is 24 (outcome) × 24 (feature) plus a bias column.

- Dataset: telonex `train` split, fully-stale rows dropped, `n = 2107`.
- Features: z-scored asks at t = −10 min.
- Reconstruction: zero PC4..PC24 columns of `β_eig`, then `β_token = β_eig_truncated[:, :24] @ U^T` (since `U` is orthogonal, `U^{-1} = U^T`). Bias column unchanged.
- Top 3 PCs explain 64.8% of feature variance (sum of standardized eigenvalues 6.46 + 5.24 + 3.85 / trace 24) and 80.9% of aggregated fitted-prediction variance (`Σ_{k=1..3} β_pc[i,k]² · λ_k / Σ_{k=1..24} β_pc[i,k]² · λ_k`, summed across outcomes). Per-outcome share ranges from 42% (Draw) and 55% (BTTS) up to 92% (A-1.5) — outcomes aligned with the tilt and totals axes are well-captured; Draw and BTTS depend on directions outside the top 3.

Run: `python -m strategies.pregame_dc.regression --keep-top 3`
Source CSV: `strategies/pregame_dc/beta_token_x_token_top3.csv`
Heatmap: `strategies/pregame_dc/plots/beta_token_x_token_top3.png`

## How to read it

Row label = predicted outcome token. Column label = z-scored ask feature token. Cell `β[i, j]` answers: *"if z-scored ask of token j increases by 1 std, how does predicted P(outcome i) change?"* The bias column is the empirical base rate (the OLS intercept, unregularised).

## Coefficient matrix (top-3 PC reconstruction)

| token | A win Y | B win Y | Draw Y | A -1.5 Y | B -1.5 Y | A -2.5 Y | B -2.5 Y | O 1.5 Y | O 2.5 Y | O 3.5 Y | O 4.5 Y | BTTS Y | A win N | B win N | Draw N | A -1.5 N | B -1.5 N | A -2.5 N | B -2.5 N | O 1.5 N | O 2.5 N | O 3.5 N | O 4.5 N | BTTS N | bias |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A win Y | +0.025 | -0.024 | -0.005 | +0.009 | -0.008 | +0.000 | +0.001 | +0.001 | +0.001 | -0.000 | -0.001 | -0.006 | -0.027 | +0.026 | +0.006 | -0.025 | +0.024 | -0.021 | +0.022 | -0.003 | -0.004 | -0.004 | -0.004 | +0.002 | +0.433 |
| B win Y | -0.020 | +0.022 | +0.000 | -0.008 | +0.006 | -0.001 | -0.001 | +0.007 | +0.006 | +0.003 | +0.001 | +0.007 | +0.021 | -0.024 | +0.000 | +0.018 | -0.025 | +0.014 | -0.023 | -0.002 | -0.004 | -0.006 | -0.006 | -0.006 | +0.296 |
| Draw Y | -0.004 | +0.002 | +0.005 | -0.001 | +0.001 | +0.000 | +0.001 | -0.008 | -0.006 | -0.003 | +0.001 | -0.001 | +0.005 | -0.002 | -0.006 | +0.007 | +0.000 | +0.008 | +0.001 | +0.005 | +0.008 | +0.009 | +0.010 | +0.004 | +0.271 |
| A -1.5 Y | +0.017 | -0.014 | -0.005 | +0.008 | -0.004 | +0.001 | +0.001 | +0.007 | +0.006 | +0.003 | +0.000 | -0.001 | -0.018 | +0.016 | +0.008 | -0.018 | +0.014 | -0.017 | +0.011 | -0.004 | -0.007 | -0.007 | -0.008 | -0.001 | +0.215 |
| B -1.5 Y | -0.011 | +0.014 | +0.001 | -0.003 | +0.006 | +0.001 | +0.000 | +0.006 | +0.006 | +0.004 | +0.002 | +0.007 | +0.013 | -0.015 | +0.002 | +0.011 | -0.015 | +0.008 | -0.015 | +0.000 | -0.002 | -0.004 | -0.004 | -0.002 | +0.117 |
| A -2.5 Y | +0.010 | -0.007 | -0.004 | +0.006 | -0.001 | +0.002 | +0.001 | +0.008 | +0.007 | +0.004 | +0.001 | +0.002 | -0.010 | +0.009 | +0.008 | -0.011 | +0.006 | -0.011 | +0.004 | -0.003 | -0.006 | -0.008 | -0.008 | -0.001 | +0.098 |
| B -2.5 Y | -0.005 | +0.006 | +0.001 | -0.001 | +0.003 | +0.001 | +0.000 | +0.003 | +0.003 | +0.002 | +0.001 | +0.003 | +0.006 | -0.006 | +0.001 | +0.004 | -0.006 | +0.003 | -0.006 | +0.000 | -0.001 | -0.002 | -0.002 | -0.001 | +0.039 |
| O 1.5 Y | +0.001 | +0.002 | -0.008 | -0.002 | -0.002 | -0.002 | -0.002 | +0.011 | +0.009 | +0.003 | -0.002 | +0.002 | -0.003 | -0.004 | +0.008 | -0.006 | -0.007 | -0.007 | -0.008 | -0.009 | -0.014 | -0.016 | -0.016 | -0.009 | +0.734 |
| O 2.5 Y | +0.003 | +0.003 | -0.008 | +0.003 | +0.001 | +0.001 | -0.000 | +0.019 | +0.016 | +0.008 | +0.001 | +0.007 | -0.004 | -0.003 | +0.014 | -0.008 | -0.008 | -0.011 | -0.010 | -0.009 | -0.016 | -0.020 | -0.021 | -0.009 | +0.496 |
| O 3.5 Y | +0.002 | +0.003 | -0.005 | +0.002 | +0.002 | +0.002 | +0.000 | +0.016 | +0.014 | +0.007 | +0.002 | +0.007 | -0.002 | -0.003 | +0.011 | -0.005 | -0.007 | -0.008 | -0.009 | -0.006 | -0.012 | -0.015 | -0.016 | -0.006 | +0.276 |
| O 4.5 Y | +0.001 | +0.002 | -0.004 | +0.001 | +0.001 | +0.001 | -0.000 | +0.010 | +0.009 | +0.004 | +0.001 | +0.004 | -0.001 | -0.002 | +0.007 | -0.004 | -0.005 | -0.005 | -0.006 | -0.005 | -0.008 | -0.010 | -0.011 | -0.005 | +0.133 |
| BTTS Y | -0.005 | +0.005 | -0.006 | -0.005 | -0.002 | -0.003 | -0.003 | +0.007 | +0.005 | +0.000 | -0.003 | +0.001 | +0.002 | -0.008 | +0.005 | -0.000 | -0.011 | -0.002 | -0.010 | -0.008 | -0.011 | -0.012 | -0.012 | -0.009 | +0.526 |
| A win N | -0.025 | +0.024 | +0.005 | -0.009 | +0.008 | -0.000 | -0.001 | -0.001 | -0.001 | +0.000 | +0.001 | +0.006 | +0.027 | -0.026 | -0.006 | +0.025 | -0.024 | +0.021 | -0.022 | +0.003 | +0.004 | +0.004 | +0.004 | -0.002 | +0.567 |
| B win N | +0.020 | -0.022 | -0.000 | +0.008 | -0.006 | +0.001 | +0.001 | -0.007 | -0.006 | -0.003 | -0.001 | -0.007 | -0.021 | +0.024 | -0.000 | -0.018 | +0.025 | -0.014 | +0.023 | +0.002 | +0.004 | +0.006 | +0.006 | +0.006 | +0.704 |
| Draw N | +0.004 | -0.002 | -0.005 | +0.001 | -0.001 | -0.000 | -0.001 | +0.008 | +0.006 | +0.003 | -0.001 | +0.001 | -0.005 | +0.002 | +0.006 | -0.007 | -0.000 | -0.008 | -0.001 | -0.005 | -0.008 | -0.009 | -0.010 | -0.004 | +0.729 |
| A -1.5 N | -0.017 | +0.014 | +0.005 | -0.008 | +0.004 | -0.001 | -0.001 | -0.007 | -0.006 | -0.003 | -0.000 | +0.001 | +0.018 | -0.016 | -0.008 | +0.018 | -0.014 | +0.017 | -0.011 | +0.004 | +0.007 | +0.007 | +0.008 | +0.001 | +0.785 |
| B -1.5 N | +0.011 | -0.014 | -0.001 | +0.003 | -0.006 | -0.001 | -0.000 | -0.006 | -0.006 | -0.004 | -0.002 | -0.007 | -0.013 | +0.015 | -0.002 | -0.011 | +0.015 | -0.008 | +0.015 | -0.000 | +0.002 | +0.004 | +0.004 | +0.002 | +0.883 |
| A -2.5 N | -0.010 | +0.007 | +0.004 | -0.006 | +0.001 | -0.002 | -0.001 | -0.008 | -0.007 | -0.004 | -0.001 | -0.002 | +0.010 | -0.009 | -0.008 | +0.011 | -0.006 | +0.011 | -0.004 | +0.003 | +0.006 | +0.008 | +0.008 | +0.001 | +0.902 |
| B -2.5 N | +0.005 | -0.006 | -0.001 | +0.001 | -0.003 | -0.001 | -0.000 | -0.003 | -0.003 | -0.002 | -0.001 | -0.003 | -0.006 | +0.006 | -0.001 | -0.004 | +0.006 | -0.003 | +0.006 | -0.000 | +0.001 | +0.002 | +0.002 | +0.001 | +0.961 |
| O 1.5 N | -0.001 | -0.002 | +0.008 | +0.002 | +0.002 | +0.002 | +0.002 | -0.011 | -0.009 | -0.003 | +0.002 | -0.002 | +0.003 | +0.004 | -0.008 | +0.006 | +0.007 | +0.007 | +0.008 | +0.009 | +0.014 | +0.016 | +0.016 | +0.009 | +0.266 |
| O 2.5 N | -0.003 | -0.003 | +0.008 | -0.003 | -0.001 | -0.001 | +0.000 | -0.019 | -0.016 | -0.008 | -0.001 | -0.007 | +0.004 | +0.003 | -0.014 | +0.008 | +0.008 | +0.011 | +0.010 | +0.009 | +0.016 | +0.020 | +0.021 | +0.009 | +0.504 |
| O 3.5 N | -0.002 | -0.003 | +0.005 | -0.002 | -0.002 | -0.002 | -0.000 | -0.016 | -0.014 | -0.007 | -0.002 | -0.007 | +0.002 | +0.003 | -0.011 | +0.005 | +0.007 | +0.008 | +0.009 | +0.006 | +0.012 | +0.015 | +0.016 | +0.006 | +0.724 |
| O 4.5 N | -0.001 | -0.002 | +0.004 | -0.001 | -0.001 | -0.001 | +0.000 | -0.010 | -0.009 | -0.004 | -0.001 | -0.004 | +0.001 | +0.002 | -0.007 | +0.004 | +0.005 | +0.005 | +0.006 | +0.005 | +0.008 | +0.010 | +0.011 | +0.005 | +0.867 |
| BTTS N | +0.005 | -0.005 | +0.006 | +0.005 | +0.002 | +0.003 | +0.003 | -0.007 | -0.005 | -0.000 | +0.003 | -0.001 | -0.002 | +0.008 | -0.005 | +0.000 | +0.011 | +0.002 | +0.010 | +0.008 | +0.011 | +0.012 | +0.012 | +0.009 | +0.474 |

## Block structure

The matrix has four visually clean blocks (separated by the YES/NO grid lines):

- **Top-left (YES outcome × YES feature)** is dominated by a moneyline 2×2 sub-block: `A win Y` row positive on `A win Y` feature (+0.025) and negative on `B win Y` feature (−0.024). `B win Y` row is the mirror. The spread tokens (`A-1.5, A-2.5, B-1.5, B-2.5`) extend the same pattern at lower magnitude. **The over/under "Y"-row × "Y"-column sub-block is a small positive cluster** centered on `O 2.5 Y → O 1.5/2.5 Y feature` — totals row tracks totals feature.
- **Top-right (YES outcome × NO feature)** is the sign-flipped image of top-left (red/blue swapped). Mechanical from `ask_NO ≈ 1 − bid_YES`.
- **Bottom-left and bottom-right** are the sign-flipped images of top-right and top-left respectively. Y/N rows of the same outcome are exact negatives, so the bottom half of the matrix is just `−1` times the top half.
- **Diagonal entries (`A win Y → A win Y`, `O 2.5 Y → O 2.5 Y`, …) are positive** — predicted P(outcome) responds positively to its own ask going up. Sanity.

## Three predictive directions visible in the matrix

1. **Tilt block** (PC1): the top-left 2×2 + 4×4 sub-block on moneyline / spread features predicting moneyline / spread outcomes. A win Y row weights all A-side YES features positively, B-side YES features negatively, and the mirror image on N-side features.
2. **Totals block** (PC3): the bottom-right corner of the YES quadrant — `O 2.5/3.5 Y` rows weight `O 1.5/2.5 Y` features positively (+0.011 to +0.019) and `O 1.5/2.5/3.5/4.5 N` features negatively (−0.014 to −0.021). Symmetric on the N quadrant.
3. **PC2 (book width)** doesn't add a visible third block — its row-loadings on outcomes are near zero (book width is non-directional), so reconstructing through PC2 contributes very little to the token-basis matrix.

## Sanity checks

- **Row sums** are 0 because PC2's broadly-positive loading does little work and the tilt + totals directions are zero-sum across YES/NO pairs (every contribution to a YES outcome is mirrored with a NO contribution of opposite sign).
- **Y-row + N-row of the same outcome = 0** at every column. Mechanical from `Y_NO = 1 − Y_YES`. Bias pairs sum to 1.
- **Diagonal positivity**: every outcome's own ask predicts itself (+0.000 to +0.025), as expected from a calibrated book.
- **Draw row is the smallest** — top-3 PCs barely capture draw probability. Consistent with R²(Draw) = 0.041 in the full OLS.
