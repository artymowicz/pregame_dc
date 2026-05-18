# Regression coefficient matrix β (unstandardized, raw asks)

OLS fit of `y = β x` where:
- `x = (asks_24, 1)` — 24 raw ask prices at t = −10 min plus a bias term
  (no z-scoring; OLS intercept absorbs the mean).
- `y` = 24-vector of binary outcomes (`V24[:, world_idx]`).
- Dataset: telonex `train` split, fully-stale rows dropped, `n = 2107`.

β is expressed in the basis:
- **rows** = canonical token basis (12 YES + 12 NO outcome positions)
- **columns** = covariance-eigenvector basis of *raw* asks
  (PC1..PC24, descending eigenvalue) plus a final `bias` column for
  the intercept.

PC eigenvalues / variance share (from `diagnostic.py --sources telonex`,
no `--standardize`):
PC1 32.8% (eig 0.29) · PC2 16.4% (0.14) · PC3 12.8% (0.11) ·
PC4 8.8% (0.08) · PC5 6.9% (0.06) · PC6 5.3% · ... · PC24 ~0%.

Source CSV: `strategies/pregame_dc/beta_token_x_pc_raw.csv`.
Heatmap: `strategies/pregame_dc/plots/beta_token_x_pc_raw.png`.

| token | PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 | PC9 | PC10 | PC11 | PC12 | PC13 | PC14 | PC15 | PC16 | PC17 | PC18 | PC19 | PC20 | PC21 | PC22 | PC23 | PC24 | bias |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A win Y | -0.017 | -0.454 | -0.024 | +0.000 | +0.060 | -0.079 | +0.110 | -0.205 | +0.173 | -0.011 | -0.157 | +0.121 | -0.043 | +0.019 | +0.386 | +0.087 | -0.171 | -0.033 | +0.008 | +0.057 | +0.365 | +0.503 | -0.046 | -0.144 | +0.064 |
| B win Y | +0.000 | +0.343 | +0.008 | +0.036 | -0.220 | +0.085 | -0.100 | +0.192 | -0.171 | -0.093 | +0.032 | +0.246 | +0.035 | +0.010 | +0.327 | -0.064 | +0.157 | -0.135 | -0.333 | -0.274 | +0.008 | -0.728 | -0.138 | +0.710 | +0.605 |
| Draw Y | +0.016 | +0.111 | +0.016 | -0.036 | +0.160 | -0.005 | -0.010 | +0.013 | -0.002 | +0.103 | +0.124 | -0.367 | +0.008 | -0.029 | -0.713 | -0.023 | +0.014 | +0.168 | +0.325 | +0.217 | -0.373 | +0.225 | +0.183 | -0.567 | +0.332 |
| A -1.5 Y | -0.011 | -0.319 | -0.027 | -0.016 | -0.042 | -0.042 | -0.006 | -0.103 | -0.077 | +0.022 | -0.131 | +0.003 | +0.074 | +0.044 | +0.164 | -0.130 | +0.144 | +0.271 | +0.166 | +0.395 | +0.619 | +0.319 | +0.481 | -0.316 | +0.098 |
| B -1.5 Y | +0.030 | +0.200 | +0.016 | +0.035 | -0.152 | +0.033 | -0.071 | +0.088 | -0.085 | -0.056 | +0.063 | +0.170 | +0.046 | +0.137 | +0.230 | -0.170 | +0.103 | +0.077 | -0.309 | -0.279 | -0.144 | -0.091 | +0.380 | +0.491 | +0.205 |
| A -2.5 Y | +0.008 | -0.197 | -0.028 | +0.003 | -0.096 | -0.002 | -0.037 | -0.012 | -0.057 | +0.022 | -0.128 | -0.059 | +0.036 | +0.176 | +0.158 | +0.126 | +0.248 | +0.228 | +0.039 | +0.534 | +0.431 | -0.149 | +0.156 | -0.565 | +0.774 |
| B -2.5 Y | +0.020 | +0.083 | +0.011 | +0.008 | -0.069 | +0.003 | -0.026 | +0.022 | -0.036 | -0.053 | -0.011 | +0.072 | +0.038 | +0.023 | +0.073 | -0.153 | +0.162 | +0.055 | -0.051 | -0.357 | -0.215 | -0.007 | +0.526 | +0.205 | -0.075 |
| O 1.5 Y | -0.052 | -0.063 | -0.023 | +0.063 | -0.274 | +0.068 | -0.016 | +0.077 | -0.242 | +0.007 | -0.142 | +0.057 | -0.088 | -0.209 | +0.524 | -0.136 | +0.486 | -0.023 | -0.369 | -0.144 | +0.297 | +0.228 | +0.314 | -0.225 | +0.624 |
| O 2.5 Y | -0.013 | -0.105 | -0.039 | +0.009 | -0.391 | +0.048 | -0.087 | +0.059 | -0.070 | -0.017 | -0.157 | +0.041 | -0.152 | +0.015 | +0.530 | +0.172 | +0.458 | -0.261 | +0.042 | -0.209 | +0.096 | -0.093 | -0.146 | +0.356 | +0.261 |
| O 3.5 Y | -0.004 | -0.063 | -0.045 | -0.009 | -0.305 | +0.034 | -0.101 | -0.004 | -0.037 | +0.032 | -0.160 | -0.042 | -0.079 | +0.042 | +0.225 | -0.012 | +0.317 | +0.105 | -0.028 | +0.044 | +0.075 | -0.127 | +0.072 | +0.695 | +0.077 |
| O 4.5 Y | -0.006 | -0.048 | -0.014 | +0.006 | -0.191 | +0.021 | -0.092 | +0.048 | -0.054 | +0.087 | -0.147 | -0.123 | -0.047 | +0.047 | +0.155 | +0.084 | +0.080 | -0.015 | -0.233 | +0.201 | -0.543 | -0.597 | -0.124 | +0.053 | +0.855 |
| BTTS Y | -0.062 | +0.034 | -0.014 | +0.036 | -0.241 | +0.089 | +0.002 | +0.098 | -0.145 | +0.088 | -0.088 | -0.040 | -0.230 | -0.328 | +0.304 | +0.084 | +0.223 | -0.327 | -0.524 | -0.287 | -0.226 | +0.130 | -0.526 | +0.167 | +0.320 |
| A win N | +0.017 | +0.454 | +0.024 | -0.000 | -0.060 | +0.079 | -0.110 | +0.205 | -0.173 | +0.011 | +0.157 | -0.121 | +0.043 | -0.019 | -0.386 | -0.087 | +0.171 | +0.033 | -0.008 | -0.057 | -0.365 | -0.503 | +0.046 | +0.144 | +0.936 |
| B win N | -0.000 | -0.343 | -0.008 | -0.036 | +0.220 | -0.085 | +0.100 | -0.192 | +0.171 | +0.093 | -0.032 | -0.246 | -0.035 | -0.010 | -0.327 | +0.064 | -0.157 | +0.135 | +0.333 | +0.274 | -0.008 | +0.728 | +0.138 | -0.710 | +0.395 |
| Draw N | -0.016 | -0.111 | -0.016 | +0.036 | -0.160 | +0.005 | +0.010 | -0.013 | +0.002 | -0.103 | -0.124 | +0.367 | -0.008 | +0.029 | +0.713 | +0.023 | -0.014 | -0.168 | -0.325 | -0.217 | +0.373 | -0.225 | -0.183 | +0.567 | +0.668 |
| A -1.5 N | +0.011 | +0.319 | +0.027 | +0.016 | +0.042 | +0.042 | +0.006 | +0.103 | +0.077 | -0.022 | +0.131 | -0.003 | -0.074 | -0.044 | -0.164 | +0.130 | -0.144 | -0.271 | -0.166 | -0.395 | -0.619 | -0.319 | -0.481 | +0.316 | +0.902 |
| B -1.5 N | -0.030 | -0.200 | -0.016 | -0.035 | +0.152 | -0.033 | +0.071 | -0.088 | +0.085 | +0.056 | -0.063 | -0.170 | -0.046 | -0.137 | -0.230 | +0.170 | -0.103 | -0.077 | +0.309 | +0.279 | +0.144 | +0.091 | -0.380 | -0.491 | +0.795 |
| A -2.5 N | -0.008 | +0.197 | +0.028 | -0.003 | +0.096 | +0.002 | +0.037 | +0.012 | +0.057 | -0.022 | +0.128 | +0.059 | -0.036 | -0.176 | -0.158 | -0.126 | -0.248 | -0.228 | -0.039 | -0.534 | -0.431 | +0.149 | -0.156 | +0.565 | +0.226 |
| B -2.5 N | -0.020 | -0.083 | -0.011 | -0.008 | +0.069 | -0.003 | +0.026 | -0.022 | +0.036 | +0.053 | +0.011 | -0.072 | -0.038 | -0.023 | -0.073 | +0.153 | -0.162 | -0.055 | +0.051 | +0.357 | +0.215 | +0.007 | -0.526 | -0.205 | +1.075 |
| O 1.5 N | +0.052 | +0.063 | +0.023 | -0.063 | +0.274 | -0.068 | +0.016 | -0.077 | +0.242 | -0.007 | +0.142 | -0.057 | +0.088 | +0.209 | -0.524 | +0.136 | -0.486 | +0.023 | +0.369 | +0.144 | -0.297 | -0.228 | -0.314 | +0.225 | +0.376 |
| O 2.5 N | +0.013 | +0.105 | +0.039 | -0.009 | +0.391 | -0.048 | +0.087 | -0.059 | +0.070 | +0.017 | +0.157 | -0.041 | +0.152 | -0.015 | -0.530 | -0.172 | -0.458 | +0.261 | -0.042 | +0.209 | -0.096 | +0.093 | +0.146 | -0.356 | +0.739 |
| O 3.5 N | +0.004 | +0.063 | +0.045 | +0.009 | +0.305 | -0.034 | +0.101 | +0.004 | +0.037 | -0.032 | +0.160 | +0.042 | +0.079 | -0.042 | -0.225 | +0.012 | -0.317 | -0.105 | +0.028 | -0.044 | -0.075 | +0.127 | -0.072 | -0.695 | +0.923 |
| O 4.5 N | +0.006 | +0.048 | +0.014 | -0.006 | +0.191 | -0.021 | +0.092 | -0.048 | +0.054 | -0.087 | +0.147 | +0.123 | +0.047 | -0.047 | -0.155 | -0.084 | -0.080 | +0.015 | +0.233 | -0.201 | +0.543 | +0.597 | +0.124 | -0.053 | +0.145 |
| BTTS N | +0.062 | -0.034 | +0.014 | -0.036 | +0.241 | -0.089 | -0.002 | -0.098 | +0.145 | -0.088 | +0.088 | +0.040 | +0.230 | +0.328 | -0.304 | -0.084 | -0.223 | +0.327 | +0.524 | +0.287 | +0.226 | -0.130 | +0.526 | -0.167 | +0.680 |

## Key differences vs the standardized run

(see `beta_matrix.md` for the z-scored version)

- **R² per token is identical** (0.137 for A win, 0.116 for B win, etc.) — OLS is invariant to invertible reparameterisation of features. Only the *coefficients* change, not the fit quality.
- **The "tilt" PC is no longer #1.** In raw covariance space the leading direction is the *scoring/blowout* factor (32.8%, eig 0.29, dominated by spread-Y tokens). The favorite-tilt axis sits at PC2 (16.4%, eig 0.14).
- **Coefficient magnitudes are scale-distorted by the eigenvalues.** A direction with eigenvalue λ in raw space carries coefficients ~1/√λ larger than the same direction in z-scored space. Concrete check:
  - Standardized PC1 (tilt, eig 6.46): A-win-Y coef = −0.063
  - Unstandardized PC2 (tilt, eig 0.14): A-win-Y coef = −0.454
  - Ratio 0.454 / 0.063 ≈ 7.2, matches √(6.46/0.14) ≈ 6.8 ✓
  Same fitted values; OLS just re-routes the magnitude through whichever scaling you give it.
- **PC1 (the dominant raw-covariance direction) carries small coefficients.** `BTTS Y −0.062`, `O 1.5 Y −0.052`, `O 2.5 Y +0.013` — the largest is 0.062. The "scoring/blowout" direction in raw space is barely informative for outcomes, even though it eats 33% of the price-variance budget. (This is consistent with PC1 in raw space being driven by which markets have high cross-sectional variance, not by which directions predict outcomes.)
- **PC2 (tilt) carries the largest moneyline/spread coefficients.** A win Y −0.454, B win Y +0.343, A-1.5 Y −0.319, A-1.5 N +0.319 — exactly the favorite-tilt structure, but on a larger absolute scale than the standardized run because the eigenvalue is small.
- **Tail PCs (PC15 onwards) carry massive coefficients** (up to ±0.7) with negligible x-variance — the OLS-on-low-variance-direction amplification is more visible here than in the standardized version since eigenvalues drop further.
- **Bias is no longer the empirical base rate.** With raw asks the OLS intercept is `y_hat at x = 0`, not at `x = mean`. Some bias entries even exceed 1 (B-2.5 N = 1.075) or fall below 0 (B-2.5 Y = −0.075). YES + NO bias pairs still sum exactly to 1.
- **Y / N rows are still exact negatives** of each other. (Mechanical from `Y_NO = 1 − Y_YES`; unchanged by feature scaling.)

## R² per token (unchanged — invariant to feature scaling)

| token | R² | base rate |
|:---|---:|---:|
| A win Y | +0.137 | 0.433 |
| B win Y | +0.116 | 0.296 |
| Draw Y | +0.041 | 0.271 |
| A -1.5 Y | +0.098 | 0.215 |
| B -1.5 Y | +0.090 | 0.117 |
| A -2.5 Y | +0.087 | 0.098 |
| B -2.5 Y | +0.054 | 0.039 |
| O 1.5 Y | +0.053 | 0.734 |
| O 2.5 Y | +0.056 | 0.496 |
| O 3.5 Y | +0.040 | 0.276 |
| O 4.5 Y | +0.037 | 0.133 |
| BTTS Y | +0.035 | 0.526 |

## Summary

The standardized version is the more useful diagnostic basis: each PC carries roughly comparable fitted-prediction contribution, so coefficient magnitudes are read directly as "informativeness". The unstandardized version distorts magnitudes by feature variance, making predictive directions with low x-variance look numerically dominant — visually the matrix is harder to read, and the bias column no longer reduces to the base rate. Same fit, same R²; just a less interpretable basis.
