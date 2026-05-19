# pregame_dc

A Dixon-Coles goal-model predictor for the outcomes of a set of 12 soccer game-related Polymarket markets (3 moneyline + 4 spread + 4 totals + 1 both teams to score), plus a live trading bot that implements a simple threshold-based trading strategy using this signal.

> This project began as `pregame_pca`, a rank-K PCR predictor; the live model
> is now Dixon-Coles (see below) and the package was renamed to `pregame_dc`.
> The PCR analysis scripts remain under `analyze/`.

## Algorithm description

Each game is modelled by two latent goal rates: team A scores $n_A \sim \mathrm{Poisson}(a)$ and team B scores $n_B \sim \mathrm{Poisson}(b)$, with a Dixon-Coles low-score correction $\tau(\rho)$ that re-weights the four 0/1-goal cells to capture the well-known draw / low-score excess. Every one of the 12 markets is a fixed region of the $(n_A, n_B)$ goal lattice, so its probability is a deterministic function of $(a, b, \rho)$.

The rates are linear in the 24 best-ask prices through a softplus link: $a = \mathrm{softplus}(F \cdot w_a)$, $b = \mathrm{softplus}(F \cdot w_b)$, where $F$ is the z-scored 24-ask vector plus a bias. The 51 parameters $(w_a, w_b, \rho)$ are fit once, offline, by L-BFGS minimising Brier loss on ~3200 telonex games at t = −10 min (`fit_save_dc.py` → `models/dc_t-10min.npz`).

The live bot implements the following strategy: at t minutes before kickoff, compute the 12 predicted probabilities. For a chosen set of markets (default = moneyline + totals), if the firing rule clears `--threshold`, place a buy order (FOK) for that token. The firing rule is either `edge = pred − ask` or `ratio = edge / max(book_spread, 0.005)` (default).

## Data sources

Train set: ~2200 games from telonex.io

Val set: ~410 games self-collected using the included script data_collection/ws_logger.py

Both datasets have the full timeseries of all 24 best asks for t=-30 minutes to t=+180 minutes relative to kickoff (although we only use the pregame part for this project)

**Data quality:** Self-collected data appears to be of good quality while telonex data has some issues, making it unreliable for validation/backtesting. This is why we use it only to train.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Train + save the live model (Dixon-Coles at t = −10 min, on full telonex).
python -m pregame_dc.live.fit_save_dc

# Run the live trading bot (dry-run by default; see Live trading below).
python -m pregame_dc.live.bot
```

## Layout

```
pregame_dc/
    paths.py                     central path constants
    constants.py                 MARKET_LABELS, X_COLS, Y_COLS, S_COLS
    analyze/                     OOS eval, calibration, plotting
    live/                        live Polymarket trading bot
        bot.py                   main asyncio Dixon-Coles bot
        fit_save_dc.py           Dixon-Coles training + .npz save
        clob_v2.py               CLOB v2 EIP-712 signing
        resolve_outcomes.py      post-game backfill of outcomes/PnL
    pipelines/                   data builders
        build_labeled_dataset.py source-parameterised labeled-dataset builder
        build_telonex_games_csv.py reconstructs data/telonex/games.csv
        helpers.py               pivot, slot mapping, outcome lookup
        self_collected/                      raw self_collected WS log → per-game per_game_data
        telonex/                 Telonex API → per-game per_game_data
    discovery/                   game discovery + slot mapping
        soccer.py                Gamma fetch + 12-slot constraint solver
        slots.py                 standard_slot_map (canonical slot ordering)
        complete_sets.py         constraint primitives
    polymarket/                  Polymarket SDK (CLOB + WS)
    models/                      Dixon-Coles model + shipped .npz
        dixon_coles.py           DC goal model: fit + predict + .npz loader

data/
    labeled/
        telonex_dataset.parquet  ~50 MB train pool (2,627 games)
        self_collected_dataset.parquet       ~9 MB eval pool (461 games)
    self_collected/per_game_data/                ~5.5 GB per-game per_game_data (1,115 files)
    telonex/per_game_data/           ~1.9 GB per-game per_game_data (2,627 files)
    telonex/cache/
        kickoff_times.json       game_date_slug → kickoff epoch (shipped, warm-start)
        market_ids.json          condition_id → Gamma numeric market_id (shipped, warm-start)
    outcomes.parquet             game-market resolution table

logs/                            live bot writes here at runtime
docs/                            findings.md, beta_matrix.md, polymarket_market_metadata.md, etc.
```

## Dataset schema

Each row in the labeled parquets is one (game_slug × snapshot_t) tick:

| column | shape | meaning |
|---|---|---|
| `game_slug` | str | game identifier |
| `source` | str | `"self_collected"` or `"telonex"` |
| `split` | str | `"train"` or `"val"` (md5-hashed slug, 80/20) |
| `seconds_since_game_start` | float | t = 0 at kickoff; samples on a fixed 30s grid over [−30 min, +180 min] |
| `x_0..x_23` | float | top-of-book asks. `x_0..x_11` = YES asks for canonical slots 0..11; `x_12..x_23` = NO asks for the same slots. |
| `s_0..s_23` | float | matching ask sizes (level-0). |
| `y_0..y_11` | int | canonical slot outcomes — `y_k = 1` if the YES side of slot k won. Constant within a game. |

Consumers reconstruct the 24-element outcome vector via
`Y = np.hstack([y, 1.0 - y])`.

The 12 canonical slots are: A win, B win, Draw, A −1.5, B −1.5, A −2.5,
B −2.5, Over 1.5, Over 2.5, Over 3.5, Over 4.5, BTTS.

## Out-of-sample evaluation

`pregame_dc/analyze/cross_eval_self_collected_dc.py` is the held-out check on
the Dixon-Coles model:

```bash
python -m pregame_dc.analyze.cross_eval_self_collected_dc
```

**What it does.** It fits a *fresh* Dixon-Coles model on the telonex games
whose slugs do **not** appear in the self_collected dataset, then scores it on
the self_collected games. ~96.5% of self_collected games are also in telonex
(same Polymarket games, different data feeds), so excluding every
self_collected slug from training makes the evaluation genuinely held out by
game identity. This is a *separate* fit from the shipped
`models/dc_t-10min.npz` — the shipped model trains on the full telonex set and
therefore cannot be honestly backtested (see [Models](#models)); this script
exists precisely to give an honest out-of-sample number.

**Apples-to-apples across data feeds.** The eval games are loaded from both
feeds and inner-joined on `game_slug`. Every metric is computed twice — once
scoring the self_collected asks, once the telonex asks — but always over the
same set of cells where *both* feeds carry a real quote (~94%). If the two
reports agree, the result is not an artefact of one feed's quirks.

**Reading the output.** A fit line reports the held-out training size, the
fitted `rho`, and the training Brier. Then, for each of the two paired reports:

- `Mean Brier (12 YES)` — mean Brier over the 12 YES markets for the base rate
  (predict the training mean) versus Dixon-Coles, plus the implied R². Lower DC
  Brier / positive R² means the model beats the base rate.
- The threshold table — the **edge firing rule** applied to the eval set: for
  each threshold, buy every token whose `pred − ask` exceeds it, paying the
  ask. Cells are `n / pnl-per-trade` with `pnl = outcome − ask`, shown as
  aggregate (`agg`) and by market type (`mny` moneyline, `spr` spread, `tot`
  totals, `btts`). This is in-eval PnL — equal-weighted unit trades, no
  transaction costs beyond the ask — so treat it as signal diagnostics, not a
  trading P&L.

A companion script, `cross_eval_best_of_dc.py`, uses the same training split
(telonex minus self_collected) but validates on the **best_of** feed — telonex
with each sentinel ask patched from self_collected where available, the
cleanest eval feed. It is held out by the same game-identity argument and
prints the same Brier / R² / threshold-table output for a single feed.

These two `cross_eval_*_dc.py` files are the DC-aware scripts under `analyze/`.
The legacy PCR scripts (`cross_eval_self_collected_clean.py`, the
`calibration_*` and `backtest_*` scripts) are unchanged and still evaluate
rank-K PCR — they are *not* DC.

## Live trading

```bash
# Dry run (default, no real orders)
python -m pregame_dc.live.bot

# Real orders (requires .env with POLYMARKET_PRIVATE_KEY + POLYMARKET_FUNDER_ADDRESS).
# Defaults: rule=ratio, threshold=4.0, markets=moneyline,totals, model=dc_t-10min.npz.
python -m pregame_dc.live.bot --live --budget 100

# After games resolve, backfill outcomes
python -m pregame_dc.live.resolve_outcomes
```

The bot:
1. Discovers upcoming soccer games via Gamma every hour
2. Subscribes to all candidate token books via the Polymarket Market WS
3. At `kickoff + --time-seconds` for each game, computes the 12 Dixon-Coles
   market probabilities from the 24 token asks
4. For each candidate slot (filtered by `--markets`, default moneyline +
   totals) where the score exceeds `--threshold` — `pred − ask` under
   `--rule=edge` or `(pred − ask) / max(book_spread, 0.005)` under
   `--rule=ratio` (default) — posts a FOK buy via the CLOB v2 endpoint

Unlike the former PCR bot there is no `res_norm` gate: Dixon-Coles has no
principal-component subspace, so every discovered game is evaluated and the
threshold alone decides what fires.

Logs — each run stamps its files with its UTC start time (`<run>` =
`YYYYMMDDThhmmssZ`), so a new run never overwrites or interleaves with an
earlier one:
- `logs/orders_summary_<run>.jsonl` — one line per attempted order
- `logs/kickoffs_<run>.jsonl` — one line per processed kickoff
- `logs/ws_events_master_<run>.jsonl` — all user-WS events
- `logs/ws_events/{slug}_{slot}_{ts}.jsonl` — per-order event tees
- `logs/stdout_<run>.log` — full console output (stdout + stderr) for the run
- `logs/orders_summary_resolved.jsonl` — `resolve_outcomes` output: every
  run's orders combined and augmented with outcomes & PnL
- `logs/state.json` — cross-run record of already-processed kickoffs (shared,
  not stamped, so a restart does not re-fire games it already handled)

## Rebuilding datasets from scratch

The shipped labeled parquets and per-game per_game_data are sufficient for
analysis and live trading. To regenerate the labeled parquets (e.g. after
adding new games to the per_game_data directories):

```bash
python -m pregame_dc.pipelines.build_labeled_dataset \
    --source self_collected --output data/labeled/self_collected_dataset.parquet
python -m pregame_dc.pipelines.build_labeled_dataset \
    --source telonex --output data/labeled/telonex_dataset.parquet
```

To rebuild raw per_game_data, see the [Collecting raw data](#collecting-raw-data) section below.

## Collecting raw data

The `data_collection/` subpackage owns everything between live external
sources and `data/{self_collected,telonex}/per_game_data/`:

```bash
# Discover today's soccer markets (writes data_collection/markets/soccer.csv)
python -m data_collection.discover_markets

# Long-running WebSocket logger that writes raw HDF5 events to data/raw_ws_logs/
# (re-runs market discovery every hour; soccer-only).
python -m data_collection.ws_logger

# One-time: download the Telonex market roster (~600 MB, anonymous endpoint).
# Required by data_collection.telonex.build_dataset below.
python -m data_collection.telonex.fetch_metadata

# Turn raw .h5 logs into per-game parquets at data/self_collected/per_game_data/.
# Also writes data/self_collected/games.csv.
python -m data_collection.self_collected.build_dataset

# Pull quote data from the Telonex API and write per-game parquets at
# data/telonex/per_game_data/. Requires TELONEX_API_KEY in .env.
# Resolves missing kickoffs via Gamma (writes data/telonex/cache/kickoff_times.json)
# and missing numeric market_ids via Gamma (writes data/telonex/cache/market_ids.json);
# both caches are merge-only and ship pre-seeded for a warm start.
python -m data_collection.telonex.build_dataset

# Reconstruct data/telonex/games.csv from the per-game parquets + telonex
# markets.parquet + the kickoff cache. Triggers the kickoff resolver if any
# per-game parquet lacks a cached kickoff.
python -m pregame_dc.pipelines.build_telonex_games_csv

# Fetch resolved-market outcomes from Gamma into data/outcomes.parquet.
# Reads market_ids from data/{self_collected,telonex}/games.csv (written by
# self_collected.build_dataset and build_telonex_games_csv respectively).
# Resumable; only fetches new/unresolved markets.
python -m data_collection.fetch_outcomes
```

Once `data/{self_collected,telonex}/per_game_data/` are populated and
`data/outcomes.parquet` covers their markets, run
`pregame_dc.pipelines.build_labeled_dataset` (above) to regenerate the
labeled parquets.

## Models

Trained model artifacts live in `pregame_dc/models/`:

- `dc_t-10min.npz` — **current live-trading model**: Dixon-Coles fit at
  t = −10 min on the full telonex dataset (train n = 3232), Brier loss.
  Produced by `python -m pregame_dc.live.fit_save_dc`.
- `rank4_t-10min.npz`, `rank3_t-*.npz` — legacy PCR models from the former
  deployment. Retained for reproducibility of older backtests; not used by
  the current bot.

The Dixon-Coles `.npz` carries `mu` (24,), `sd_safe` (24,), `w_a` (25,),
`w_b` (25,), `rho` (scalar), and provenance fields `loss`, `T_TARGET`,
`train_n`. The bot loads it via
`pregame_dc.models.dixon_coles.DixonColesModel`.

Note: the shipped model trains on the *full* telonex set, so it cannot be
honestly backtested against games it has seen. An out-of-sample backtest
must fit a separate model with the evaluation games excluded (e.g.
`fit_save_dc.py --exclude-self-collected`).
