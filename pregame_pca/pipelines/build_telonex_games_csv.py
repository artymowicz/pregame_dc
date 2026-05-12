"""Reconstruct data/telonex/games.csv (h5-compatible schema) from per-game data.

Joins:
  - per-game parquets in data/telonex/per_game_data/ (market_id, condition_id,
    type, question already baked in)
  - data/telonex/metadata/markets.parquet for event_slug + yes/no tokens
  - data/telonex/cache/kickoff_times.json for game_start_time

If any per-game parquet has no kickoff cached, the script will invoke the
telonex kickoff resolver to fetch the missing entries (which appends to the
cache without overwriting existing data) before building the CSV.

Output columns (matches the h5/pmxt_v2 games.csv schema used elsewhere):
    game_slug, game_id, game_start_time, event_slug, market_id, condition_id,
    type, question, yes_token, no_token

Usage:
    python -m pregame_pca.pipelines.build_telonex_games_csv
"""
from __future__ import annotations

import asyncio
import csv
import json
import sys
from datetime import datetime, timezone

import pyarrow.parquet as pq

from data_collection.telonex.build_dataset import resolve_kickoffs
from pregame_pca import paths
from pregame_pca.polymarket.soccer import parse_game_date_slug, parse_game_slug


def load_markets_index():
    """Map condition_id (hex) -> dict{event_slug, asset_id_0, asset_id_1, question}."""
    print(f"[markets] loading {paths.TELONEX_METADATA} ...", file=sys.stderr, flush=True)
    tbl = pq.read_table(paths.TELONEX_METADATA,
                        columns=["market_id", "event_slug", "question",
                                 "asset_id_0", "asset_id_1"])
    df = tbl.to_pandas()
    df = df[df["market_id"].astype(str).str.startswith("0x")].copy()
    out = {}
    for _, r in df.iterrows():
        out[r["market_id"]] = {
            "event_slug": r["event_slug"] or "",
            "asset_id_0": r["asset_id_0"] or "",
            "asset_id_1": r["asset_id_1"] or "",
            "question": r["question"] or "",
        }
    print(f"[markets] indexed {len(out):,} markets", file=sys.stderr)
    return out


def load_kickoffs():
    if not paths.TELONEX_KICKOFF_CACHE.exists():
        return {}
    with open(paths.TELONEX_KICKOFF_CACHE) as f:
        raw = json.load(f)
    return {gds: datetime.fromtimestamp(ts, tz=timezone.utc) for gds, ts in raw.items()}


def fmt_kickoff(dt):
    # Match the existing h5/pmxt_v2 style: 'YYYY-MM-DD HH:MM:SS+00'
    return dt.strftime("%Y-%m-%d %H:%M:%S+00")


def resolve_missing_kickoffs(snap_paths, markets_idx, kickoffs):
    """For per-game parquets with no cached kickoff, call resolve_kickoffs.

    resolve_kickoffs writes back to the kickoff cache by merging onto the
    existing memo, so previously-resolved entries are never lost.
    """
    missing = [p for p in snap_paths if p.stem not in kickoffs]
    if not missing:
        return kickoffs
    print(f"[kickoffs] {len(missing)} per-game parquets have no cached "
          f"kickoff; running resolver ...", file=sys.stderr, flush=True)

    games_for_resolver = {}
    for p in missing:
        df = pq.read_table(p, columns=["condition_id"]).to_pandas()
        cids = df["condition_id"].drop_duplicates().tolist()
        markets = []
        for cid in cids:
            meta = markets_idx.get(cid)
            if not meta:
                continue
            markets.append({"condition_id": cid,
                            "event_slug": meta["event_slug"]})
        if markets:
            games_for_resolver[p.stem] = markets

    if not games_for_resolver:
        print("[kickoffs] no resolvable per-game parquets (all condition_ids "
              "missing from metadata); skipping resolver", file=sys.stderr)
        return kickoffs

    asyncio.run(resolve_kickoffs(games_for_resolver))
    return load_kickoffs()


def main():
    markets_idx = load_markets_index()
    kickoffs = load_kickoffs()
    print(f"[kickoffs] {len(kickoffs):,} game_date_slugs", file=sys.stderr)

    snap_paths = sorted(paths.TELONEX_SNAPSHOTS.glob("*.parquet"))
    print(f"[per_game_data] {len(snap_paths):,} game parquet files", file=sys.stderr)

    kickoffs = resolve_missing_kickoffs(snap_paths, markets_idx, kickoffs)

    rows = []
    n_missing_kickoff = 0
    n_missing_metadata = 0

    for i, p in enumerate(snap_paths, 1):
        game_date_slug = p.stem  # e.g. arg-aae-riv-2026-03-22
        game_slug = parse_game_slug(game_date_slug)
        kickoff_dt = kickoffs.get(game_date_slug)
        if kickoff_dt is None:
            n_missing_kickoff += 1
            continue
        kickoff_str = fmt_kickoff(kickoff_dt)

        # Read per-market identifiers from the per-game parquet
        df = pq.read_table(p, columns=["market_id", "condition_id", "type", "question"]).to_pandas()
        df = df.drop_duplicates("condition_id")

        for _, r in df.iterrows():
            cid = r["condition_id"]
            meta = markets_idx.get(cid, {})
            if not meta:
                n_missing_metadata += 1
                continue
            ev_slug = meta["event_slug"]
            if ev_slug and parse_game_date_slug(ev_slug) != game_date_slug:
                # markets.parquet picked a different event_slug than expected; keep it anyway
                pass
            rows.append({
                "game_slug": game_slug,
                "game_id": "",
                "game_start_time": kickoff_str,
                "event_slug": ev_slug,
                "market_id": int(r["market_id"]),
                "condition_id": cid,
                "type": r["type"],
                "question": r["question"] or meta["question"],
                "yes_token": meta["asset_id_0"],
                "no_token": meta["asset_id_1"],
            })

        if i % 250 == 0:
            print(f"  [{i}/{len(snap_paths)}] {len(rows):,} rows so far",
                  file=sys.stderr, flush=True)

    print(f"[done] {len(rows):,} rows, "
          f"missing kickoff={n_missing_kickoff}, missing metadata={n_missing_metadata}",
          file=sys.stderr)

    paths.TELONEX_GAMES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.TELONEX_GAMES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "game_slug", "game_id", "game_start_time", "event_slug",
            "market_id", "condition_id", "type", "question",
            "yes_token", "no_token",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {paths.TELONEX_GAMES_CSV}  ({len(rows):,} rows, "
          f"{len(set(r['game_slug'] for r in rows)):,} unique game_slugs)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
