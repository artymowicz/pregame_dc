# pregame_pca

A rank-3 PCR (Principal Components Regression) predictor for the outcomes of a set of 12 soccer game-related Polymarket markets (3 moneyline + 4 spread + 4 totals + 1 both teams to score), plus a live trading bot that implements a simple threshold-based trading strategy using this signal.

## Algorithm description

For a given game, let $p_1,...,p_{24}$ be the best ask prices for YES and NO tokens of the 12 markets at t minutes before kickoff (default t=10). Using a dataset of ~2200 games, computes the top k (default k=3) standardized principal eigenvectors $v_1,...,v_k$. Then trains a simple linear regression of the outcome probability of the 12 markets as linear functions of the full price vector projected onto $v_1,...,v_k$.

The live bot implements the following strategy: at t minutes before kickoff, compute the predicted probabilities. For a chosen set of markets (default = moneyline only, I think), if the predicted probability is greater than the best ask price + threshold (default = 4 or 5 cents, I think), place a buy order (FOK) for that token.

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

# Reproduce the OOS R² ≈ +3.8% result at three pregame per_game_data
python -m pregame_pca.analyze.cross_eval_self_collected_multi_t

# Calibration plots
python -m pregame_pca.analyze.calibration_plots
python -m pregame_pca.analyze.calibration_plots_per_market

# Train + save model parameters (defaults: K=4 PCR at t = −10 min, on full telonex)
python -m pregame_pca.live.fit_save_model
```

## Layout

```
pregame_pca/
    paths.py                     central path constants
    constants.py                 MARKET_LABELS, X_COLS, Y_COLS, S_COLS
    analyze/                     OOS eval, calibration, plotting
    live/                        live Polymarket trading bot
        bot.py                   main asyncio bot
        fit_save_model.py        rank-3 PCR training + .npz save
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
    models/                      shipped trained model parameters (.npz)

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

## Live trading

```bash
# Dry run (default, no real orders)
python -m pregame_pca.live.bot

# Real orders (requires .env with POLYMARKET_PRIVATE_KEY + POLYMARKET_FUNDER_ADDRESS).
# Defaults: rule=ratio, threshold=1.0, res-norm-min=2.25, model=rank4_t-10min.npz.
python -m pregame_pca.live.bot --live --budget 100

# After games resolve, backfill outcomes
python -m pregame_pca.live.resolve_outcomes
```

The bot:
1. Discovers upcoming soccer games via Gamma every hour
2. Subscribes to all candidate token books via the Polymarket Market WS
3. At `kickoff + --time-seconds` for each game, computes the rank-K
   prediction (default K=4) over the 24 token asks and the residual-norm of
   the z-scored ask vector against the top-K eigenvector subspace
4. If `res_norm < --res-norm-min`, skips the whole game (the snapshot sits
   too close to the "on-manifold" subspace where the strategy is unprofitable)
5. Otherwise for each candidate slot (filtered by `--markets`) where the
   score exceeds `--threshold` — `pred − ask` under `--rule=edge` or
   `(pred − ask) / max(book_spread, 0.005)` under `--rule=ratio` (default) —
   posts a FOK buy via the CLOB v2 endpoint

Logs:
- `logs/orders_summary.jsonl` — one line per attempted order
- `logs/orders_summary_resolved.jsonl` — augmented with outcomes & PnL
- `logs/ws_events_master.jsonl` — all user-WS events
- `logs/ws_events/{slug}_{slot}_{ts}.jsonl` — per-order event teеs

## Rebuilding datasets from scratch

The shipped labeled parquets and per-game per_game_data are sufficient for
analysis and live trading. To regenerate the labeled parquets (e.g. after
adding new games to the per_game_data directories):

```bash
python -m pregame_pca.pipelines.build_labeled_dataset \
    --source self_collected --output data/labeled/self_collected_dataset.parquet
python -m pregame_pca.pipelines.build_labeled_dataset \
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
python -m pregame_pca.pipelines.build_telonex_games_csv

# Fetch resolved-market outcomes from Gamma into data/outcomes.parquet.
# Reads market_ids from data/{self_collected,telonex}/games.csv (written by
# self_collected.build_dataset and build_telonex_games_csv respectively).
# Resumable; only fetches new/unresolved markets.
python -m data_collection.fetch_outcomes
```

Once `data/{self_collected,telonex}/per_game_data/` are populated and
`data/outcomes.parquet` covers their markets, run
`pregame_pca.pipelines.build_labeled_dataset` (above) to regenerate the
labeled parquets.

## Models

Trained model artifacts in `pregame_pca/models/`:

- `rank4_t-10min.npz` — **current recommended live-trading model**: K=4 PCR
  fit at t = −10 min on the full telonex dataset (train n = 3232). Used in
  conjunction with the `res_norm > 2.25` and `edge/book_spread > 1.0` firing
  rule (see `pregame_pca/live/bot.py`).
- `rank3_t-10min.npz`, `rank3_t-25min.npz` — legacy K=3 models (former
  deployment). Retained for reproducibility of older backtests.

Each `.npz` carries `mu` (24,), `sd_safe` (24,), `beta_K` (24, 25),
`U` (24, 24), `eigvals` (24,), and provenance scalars `T_TARGET`, `K`,
`train_n`. The bot reads `mu`, `sd_safe`, `beta_K`, `U` at inference (the
last for computing `res_norm`).
