"""Build the labeled dataset for one source (self_collected or telonex).

Output schema (one row per game × 30s per_game_data tick):
    game_slug, source, split, seconds_since_game_start,
    x_0..x_23, s_0..s_23, y_0..y_11

Sampling: fixed-grid 30s ticks over [-30min, +180min] around each game's
kickoff. Filters: games must have all 12 canonical slots and fully resolved
outcomes. Train/val split: per-slug md5 hash → train_frac / (1-train_frac).

Usage:
    python -m pregame_dc.pipelines.build_labeled_dataset \\
        --source self_collected --output data/labeled/self_collected_dataset.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from pregame_dc import paths
from pregame_dc.pipelines.helpers import (
    PRE_GAME_MIN, POST_GAME_MIN,
    collect_games, load_per_game_data, pivot_per_game_data, split_by_hash,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["self_collected", "telonex"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--bucket-s", type=float, default=30.0)
    ap.add_argument("--per_game_data-per-game", type=int, default=0)
    args = ap.parse_args()

    source = args.source
    games_csv = paths.source_games_csv(source)
    snap_dir = paths.source_per_game_data_dir(source)
    print(f"collecting {source} games from {games_csv} ...")
    games = collect_games(games_csv)
    print(f"  {len(games)} eligible {source} games")

    rows = []
    n_no_snap = n_empty = n_kept = 0
    n_total = len(games)
    t0 = last_report = time.time()
    REPORT_EVERY_S = 5.0

    for i, (slug, g) in enumerate(games.items()):
        snap_path = snap_dir / f"{slug}.parquet"
        if not snap_path.exists():
            alt = list(snap_dir.glob(f"{slug}*.parquet"))
            if not alt:
                n_no_snap += 1
                continue
            snap_path = alt[0]
        snap = load_per_game_data(source, snap_path)
        if snap is None or snap.empty:
            n_empty += 1
            continue

        # Re-key snap.market_id via condition_id to match g["slot_mids"]
        # (which came from games.csv). Polymarket renumbers market_id silently
        # over time; condition_id is the stable identity. Without this, ~185
        # games hit silent slot mismatches in _pivot_one.
        csv_sub = g["games_subset"]
        mid_map = dict(zip(csv_sub["condition_id"],
                           csv_sub["market_id"].astype("int64")))
        snap = snap.assign(market_id=snap["condition_id"].map(mid_map))
        snap = snap[snap["market_id"].notna()].copy()
        snap["market_id"] = snap["market_id"].astype("int64")
        if snap.empty:
            n_empty += 1
            continue

        gst_ms = int(g["game_start_ts"]) * 1000
        sampled = pivot_per_game_data(
            snap, g["slot_mids"],
            n_samples=args.per_game_data_per_game,
            bucket_s=args.bucket_s,
            grid_start_ms=gst_ms - PRE_GAME_MIN * 60_000,
            grid_end_ms=gst_ms + POST_GAME_MIN * 60_000,
        )
        if sampled.empty:
            n_empty += 1
            continue

        sampled["game_slug"] = slug
        sampled["source"] = source
        sampled["split"] = split_by_hash(slug, args.train_frac)
        sampled["seconds_since_game_start"] = (
            sampled["timestamp_ms"] / 1000.0 - g["game_start_ts"]
        )
        # Outcomes: 12 binary y_k columns, constant within game.
        for k, v in enumerate(g["slot_outcomes"]):
            sampled[f"y_{k}"] = int(v)
        sampled = sampled.drop(columns=["timestamp_ms"])
        rows.append(sampled)
        n_kept += 1

        now = time.time()
        if now - last_report >= REPORT_EVERY_S:
            elapsed = now - t0
            done = i + 1
            rate = done / elapsed if elapsed > 0 else 0
            eta_min = (n_total - done) / rate / 60 if rate > 0 else float("inf")
            total_rows = sum(len(r) for r in rows)
            print(f"  [{done:>5}/{n_total}] kept={n_kept}  "
                  f"skipped(no_snap={n_no_snap}, empty={n_empty})  "
                  f"rows={total_rows:,}  rate={rate:.1f}/s  "
                  f"elapsed={elapsed/60:.1f}min  eta={eta_min:.1f}min", flush=True)
            last_report = now

    elapsed = time.time() - t0
    print(f"\nfinished {n_total} games in {elapsed/60:.1f}min: "
          f"kept={n_kept}  no_snap={n_no_snap}  empty_pivot={n_empty}")
    if not rows:
        raise SystemExit("produced 0 rows")

    out_df = pd.concat(rows, ignore_index=True)
    col_order = (
        ["game_slug", "source", "split", "seconds_since_game_start"]
        + [f"x_{i}" for i in range(24)]
        + [f"s_{i}" for i in range(24)]
        + [f"y_{i}" for i in range(12)]
    )
    out_df = out_df[col_order]
    print(f"\nfinal: {len(out_df):,} rows  {out_df['game_slug'].nunique()} games")
    print(out_df["split"].value_counts())
    print(f"NaN count: {out_df.isna().sum().sum()}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
