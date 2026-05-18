# pregame_dc — findings

## Setup

- **Vector**: 24-dim ask-price vector at t = −10 min before kickoff
  (`seconds_since_game_start = -600`, the 30s grid bucket).
- **Layout**: `x_0..x_11` = YES asks for canonical slots 0..11
  (A win, B win, Draw, A-1.5, B-1.5, A-2.5, B-2.5, O1.5, O2.5, O3.5, O4.5,
  BTTS); `x_12..x_23` = NO asks, same slot order.
- **Sample**: telonex `train` split, fully-stale rows dropped (every ask = 1.0).
  **n = 2,107 games**. (self_collected was excluded after 2026-05-01 because of a known
  pipeline bug being fixed in a parallel session.)
- **Script**: `python -m strategies.pregame_dc.diagnostic
  [--standardize] [--sources {self_collected,telonex}...] [--log-prices]`
- **Plots**: `strategies/pregame_dc/plots/{covariance,correlation,scree,top_eigenvectors,top_eigenvectors_heatmap}{,_telonex,_log,_std}.png`
  (combinations of source / log / std flags get appended into the suffix)

## Per-feature stats at t = −10 (telonex)

- mean range over the 24 asks: [0.261, 0.965]
- std  range: [0.051, 0.388]
- Spread-YES tokens (`A-1.5 Y, B-1.5 Y, A-2.5 Y, B-2.5 Y`) and `O 3.5/4.5 Y`
  carry the most cross-sectional variance; many moneyline + chalk tokens
  are pinned near 0 or 1 with std ~0.05.
- In log space: per-feature std range expands to [0.060, 1.439] because
  near-zero asks (chalk markets — the heavy A-favorite makes `A win N`
  → 0) get massively amplified.

## Unstandardized PCA, raw asks (covariance matrix, self_collected+telonex)

> Historical run on the combined self_collected+telonex sample (n=2439). Numbers below
> are the original combined run; telonex-only deviates by less than 0.5pp
> per PC since telonex dominates the sample.

Trace 0.825, effective rank ≈ 8.7. 5 PCs ≈ 78% of variance, 8 PCs ≈ 90%.

| PC | %var | label                            | dominant loadings (top-3)              |
|----|------|----------------------------------|----------------------------------------|
| 1  | 32.3 | scoring / blowout factor         | B-2.5 Y +0.47, A-2.5 Y +0.44, B-1.5 Y +0.40 |
| 2  | 17.2 | A-vs-B favorite tilt             | A win N +0.43, A win Y −0.41, B win Y +0.36 |
| 3  | 12.8 | residual B-blowout (axis-aligned)| B-2.5 Y +0.87                          |
| 4  |  8.6 | residual A-blowout (axis-aligned)| A-2.5 Y +0.84                          |
| 5  |  7.4 | under/over residual              | O 2.5 N +0.40, O 1.5 N +0.38, O 3.5 N +0.34 |

Driver: directions are dominated by the highest-variance tokens. PC3/PC4
are nearly pure single-token axes (`B-2.5 Y`, `A-2.5 Y`) — those tokens
carry idiosyncratic variance not predicted by the other components.

## Standardized PCA on raw asks (correlation matrix, telonex-only)

Each coordinate divided by its std before SVD (trace = 24). Effective
rank ≈ 9.95. 3 PCs ≈ 65%, 8 PCs ≈ 82%.

**Sharp elbow at PC 3 → PC 4** (eigvals 3.85 → 1.05). First three
components are the structural factors; from PC 4 onward eigvals sit near
the unit-variance noise floor.

| PC | %var | label                            | dominant loadings (top-3)              |
|----|------|----------------------------------|----------------------------------------|
| 1  | 26.9 | A-vs-B favorite tilt             | A-1.5 N +0.37, A win N +0.36, A-2.5 N +0.33 |
| 2  | 21.8 | overall ask level / book width   | A-1.5 Y +0.33, BTTS N +0.29, O 3.5 Y +0.28 (broadly +) |
| 3  | 16.1 | total goals (over/under)         | O 1.5 Y +0.38, O 2.5 Y +0.35, O 4.5 N −0.34 |
| 4  |  4.4 | blowout-magnitude residual       | B-2.5 Y +0.60, A-2.5 Y +0.47, B-1.5 Y +0.25 |
| 5  |  3.9 | definite-winner / no-BTTS        | Draw N +0.65, BTTS Y −0.41, Draw Y +0.32    |

PC1 and PC2 swap dominance vs the unstandardized run because once every
coordinate is unit-variance the previously-amplified spread-Y block stops
dominating the leading direction. PC3 becomes a much cleaner totals axis.

**Removing self_collected has essentially no effect.** The self_collected+telonex combined run
gave PC1=27.2 / PC2=21.5 / PC3=16.8% — every PC moves by less than 1pp
when self_collected is dropped, and the eigenvector loadings are visually
indistinguishable. Sample-size dominance (telonex 86% of combined) was
already swamping the self_collected signal.

## Standardized PCA on log-prices (telonex-only)

Same standardize-then-PCA pipeline but on `log(clip(ask, 1¢, 1))` instead
of raw `ask`. Trace = 24; effective rank ≈ 9.43; 3 PCs ≈ 66%, 8 PCs ≈ 85%.

**Eigenvalue gap is shallower** — λ₃→λ₄ is 3.97 → 1.39 (vs 3.85 → 1.05
on raw). Log-prices push more variance into PC 4–5 because chalk tokens
(near 0) get amplified, giving the "blowout" and "draw structure" axes
larger eigenvalues.

| PC | %var | label                              | dominant loadings (top-3)               |
|----|------|------------------------------------|-----------------------------------------|
| 1  | 29.5 | A-vs-B favorite tilt (sharper)     | A win N +0.35, A-1.5 N +0.35, A-2.5 N +0.31 |
| 2  | 19.9 | "all NO asks elevated" / chalk-N   | B-1.5 N +0.34, B win N +0.32, B-2.5 N +0.31 |
| 3  | 16.5 | "all YES asks elevated" / over     | O 3.5 Y +0.38, BTTS Y +0.38, O 4.5 Y +0.36 |
| 4  |  5.8 | draw-structure mix                 | Draw N +0.46, BTTS N −0.37, A win Y +0.30   |
| 5  |  4.6 | definite-winner / no-BTTS          | Draw N +0.63, BTTS N +0.37, O 4.5 N +0.33   |

**What changed in log space:**

- **PC1 sharpens** (26.9 → 29.5%). Log-asymmetry between near-0 and
  near-1 makes the favorite/underdog distinction more dominant — when a
  game has a heavy favorite, the underdog's NO-side asks swing in log
  far more than the favorite's, so the tilt direction explains more
  *log* variance than *raw* variance.
- **PC2 changes character.** In raw space PC2 was a near-uniform +
  vector ("all asks elevated"). In log space it splits into two
  separate factors: a "NO-side elevation" mode (PC2) and a "YES-side
  elevation" mode (PC3). The signed YES/NO splitting is the
  log-transform's way of expressing the same "book-width" idea while
  acknowledging that NO and YES asks live on different magnitude scales
  (NO is usually small for chalk markets, so log-NO is more variable
  than log-YES).
- **PC3 becomes "all over/BTTS YES asks" rather than a clean
  over/under axis.** In raw space PC3 had clean negative loadings on
  under/`Ox.5 N`; in log space those negatives are nearly zero and the
  axis is one-sided. Same physical factor but expressed differently
  because of the log-asymmetry.
- **PC4–PC5 carry more of the variance.** In raw, PC4+PC5 = 8.3% of
  variance; in log, PC4+PC5 = 10.4%. The "clean draw / shutout"
  structure (Draw N high, BTTS Y or N low) shows up earlier and carries
  more weight.

**Bottom line:** log-prices and raw-prices give the same broad story
(tilt → book level → totals → blowout → draw-structure), but log
re-balances the basis: it sharpens the tilt axis, splits the
"all-elevated" mode into YES- and NO-half versions, and gives draw /
chalk structure more share of the spectrum. If the downstream use is a
linear model whose features should be scale-invariant in the
multiplicative sense (e.g. log-odds), the log-PCA basis is the
appropriate one. If you care about additive ask movement (raw $
exposure), use the raw-PCA basis.

## Takeaways

- Pregame cross-sectional structure has **3 dominant correlation
  factors**, robust across data source (self_collected vs telonex) and feature
  encoding (raw vs log): (1) favorite tilt, (2) book level / NO-side
  elevation, (3) total goals / YES-side elevation.
- The book is roughly rank-3 in the *correlation* sense and rank ≈ 5–8 in
  the *raw* sense (extra "factors" in unstandardized PCA are really just
  the highest-variance tokens being amplified).
- Standardized PCA is the right basis for interpreting structure;
  unstandardized PCA is the right basis if a downstream loss is
  proportional to absolute price movement (e.g. trade PnL is
  variance-weighted).
- Log-price PCA sharpens tilt and gives chalk/draw structure more weight;
  raw-price PCA is more balanced across magnitude regimes.
- `B-2.5 Y` and `A-2.5 Y` carry meaningful idiosyncratic variance
  (axis-aligned PCs in unstandardized run) — possibly informative for
  blowout-direction signals.

## Open questions / next steps

- Repeat at multiple time cuts (t = −30, −10, 0, +30 min) — does the
  3-factor structure persist, sharpen, or rearrange around kickoff?
- Project games onto (PC1, PC2, PC3) and color by `world_idx` (resolved
  outcome) — do the factor axes carry resolution signal?
- Pre-residualize asks against PC1/PC2/PC3 and ask whether the residual
  is what existing MLP models are mostly fitting (or vice versa).
