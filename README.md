# pregame_pca

A rank-3 PCR (Principal Components Regression) predictor for the 12 pregame
soccer markets that Polymarket lists per game (moneyline / spread / totals /
BTTS), plus a live trading bot that fires FOK buys when the model's
prediction exceeds the current ask by a configurable threshold.

The model is fit on z-scored ask vectors from ~2,200 telonex games and
evaluated out-of-sample on ~410 self_collected games (held out by game identity). At
**t = −10 min, threshold 0.05** the moneyline edge is +13¢/share with the
test calibration well within ±2σ.

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

# Train + save model parameters at t = −10 min
python -m pregame_pca.live.fit_save_model \
    --time-seconds -600 \
    --out pregame_pca/models/rank3_t-10min.npz
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
    outcomes.parquet             game-market resolution table

logs/                            live bot writes here at runtime
docs/                            findings.md, beta_matrix.md, etc.
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

# Real orders (requires .env with POLYMARKET_PRIVATE_KEY + POLYMARKET_FUNDER_ADDRESS)
python -m pregame_pca.live.bot --live --budget 100 \
    --threshold 0.04 --markets moneyline --time-seconds -600 \
    --model pregame_pca/models/rank3_t-10min.npz

# After games resolve, backfill outcomes
python -m pregame_pca.live.resolve_outcomes
```

The bot:
1. Discovers upcoming soccer games via Gamma every hour
2. Subscribes to all candidate token books via the Polymarket Market WS
3. At `kickoff + --time-seconds` for each game, computes the rank-3
   prediction over the 24 token asks
4. For each candidate slot (filtered by `--markets`) where
   `pred − ask > --threshold`, posts a FOK buy via the CLOB v2 endpoint

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

# Turn raw .h5 logs into per-game parquets at data/self_collected/per_game_data/
python -m data_collection.self_collected.build_dataset

# Pull quote data from the Telonex API (needs TELONEX_API_KEY in .env) and
# write per-game parquets at data/telonex/per_game_data/.
python -m data_collection.telonex.build_dataset

# Fetch resolved-market outcomes from Gamma into data/outcomes.parquet.
# Reads market_ids from data/{self_collected,telonex}/games.csv (written by the
# build_dataset scripts above). Resumable; only fetches new/unresolved markets.
python -m data_collection.fetch_outcomes
```

Once `data/{self_collected,telonex}/per_game_data/` are populated and
`data/outcomes.parquet` covers their markets, run
`pregame_pca.pipelines.build_labeled_dataset` (above) to regenerate the
labeled parquets.

## Models

Two trained model artifacts are shipped in `pregame_pca/models/`:

- `rank3_t-25min.npz` — fit at t = −25 min (matches the original "fire at
  -25, threshold 0.05" deployment). Train n = 2166.
- `rank3_t-10min.npz` — fit at t = −10 min. Train n = 2170. The current
  recommended live-trading model: moneyline edge is materially larger near
  kickoff and the per-feature standardisation is more stable past −10 min.

Each `.npz` carries `mu` (24,), `sd_safe` (24,), `beta_K` (24, 25),
`U` (24, 24), `eigvals` (24,), and provenance scalars `T_TARGET`, `K`,
`train_n`. The bot only reads `mu`, `sd_safe`, `beta_K` at inference.
