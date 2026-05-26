#!/usr/bin/env python3
"""Build per-game soccer per_game_data from telonex.io `quotes` channel.

Outputs:
    data/telonex/per_game_data/{game_date_slug}.parquet
        — self_collected-compatible schema (market_id, condition_id, type, question, is_yes,
          timestamp_ms, best_bid, best_ask, best_bid_size, best_ask_size)
    data/telonex/cache/kickoff_times.json
        — game_date_slug -> kickoff_epoch_seconds (memoized across runs)
    data/telonex/cache/skipped_unknown_kickoff.txt
        — games skipped because no kickoff was resolvable

Streaming architecture: per-game, downloads ≤12 YES-side `quotes` files
concurrently into a tmpdir, clips each to [last-update-before-window,
kickoff+3.5h] so the last pre-window quote seeds downstream ffill,
appends to an in-memory game buffer, and deletes the raw file immediately.
When all markets are processed, synthesizes NO rows via YES/NO complementarity
and writes the per-game parquet. Peak disk ≈ `parallel × avg_file_size`
(≈ 3 MB at --parallel=10).

Usage:
    python -m pipelines.telonex.build_dataset --games aus-vic-new-2026-04-17 --dry-run
    python -m pipelines.telonex.build_dataset --start 2026-04-10 --end 2026-04-17
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import tempfile
import time
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pregame_dc.polymarket.soccer import classify_market_type, parse_game_date_slug

from data_collection.telonex.client import TelonexError, download_quotes

GAMMA_API = "https://gamma-api.polymarket.com"

_DATA_DIR = Path(project_root) / "data" / "telonex"
METADATA_PATH = _DATA_DIR / "metadata" / "markets.parquet"
SNAPSHOTS_DIR = _DATA_DIR / "per_game_data"
CACHE_DIR = _DATA_DIR / "cache"
KICKOFF_CACHE_PATH = CACHE_DIR / "kickoff_times.json"
MARKET_ID_CACHE_PATH = CACHE_DIR / "market_ids.json"
SKIPPED_PATH = CACHE_DIR / "skipped_unknown_kickoff.txt"

SELF_COLLECTED_GAMES_CSV = Path(project_root) / "data" / "self_collected" / "games.csv"

# Canonical full-set distribution: 3 moneyline + 4 spread + 4 totals + 1 btts.
FULL_SET_COUNTS = {"moneyline": 3, "spreads": 4, "totals": 4, "both_teams_to_score": 1}


# ---------------------------------------------------------------------------
# Target discovery from telonex markets.parquet
# ---------------------------------------------------------------------------

def _row_has_tag(tags, tag):
    if tags is None:
        return False
    for t in tags:
        if t == tag:
            return True
    return False


def discover_games(markets_path, start_date=None, end_date=None, games_filter=None,
                   league_filter=None, full_set_only=True):
    """Read markets.parquet, return {game_date_slug: [market_dict, ...]}.

    Each market_dict: {slug, event_slug, question, type, asset_id_yes,
                       condition_id, date}
    """
    print(f"[discover] Reading {markets_path} ...", file=sys.stderr, flush=True)
    t0 = time.time()

    cols = ["slug", "event_slug", "question", "tags",
            "asset_id_0", "asset_id_1", "market_id", "status"]
    table = pq.read_table(markets_path, columns=cols)
    df = table.to_pandas()

    # Filter to soccer
    df = df[df["tags"].apply(lambda x: _row_has_tag(x, "soccer"))].copy()
    print(f"[discover] {len(df):,} soccer market rows ({time.time()-t0:.1f}s)",
          file=sys.stderr)

    # Optional league filter — require any tag in the league list
    if league_filter:
        league_set = set(league_filter)
        df = df[df["tags"].apply(
            lambda ts: ts is not None and any(t in league_set for t in ts)
        )].copy()
        print(f"[discover]   {len(df):,} after league filter {sorted(league_set)}",
              file=sys.stderr)

    # Classify market type
    df["type"] = df["question"].apply(lambda q: classify_market_type(q) or "")

    # Keep only the 4 core types
    core = {"moneyline", "spreads", "totals", "both_teams_to_score"}
    df = df[df["type"].isin(core)].copy()

    # Compute game_date_slug
    df["game_date_slug"] = df["event_slug"].apply(
        lambda s: parse_game_date_slug(s) if isinstance(s, str) else ""
    )
    df = df[df["game_date_slug"] != ""].copy()

    # Parse date from slug (trailing YYYY-MM-DD)
    df["date"] = df["game_date_slug"].str.extract(r"(\d{4}-\d{2}-\d{2})$")
    df = df[df["date"].notna()].copy()

    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]
    if games_filter:
        df = df[df["game_date_slug"].isin(games_filter)]

    print(f"[discover]   {len(df):,} rows after date/games filter; grouping...",
          file=sys.stderr)

    games = {}
    for gds, sub in df.groupby("game_date_slug"):
        # Full-set check
        counts = sub["type"].value_counts().to_dict()
        if full_set_only:
            ok = all(counts.get(k, 0) >= v for k, v in FULL_SET_COUNTS.items())
            if not ok:
                continue

        markets = []
        for _, row in sub.iterrows():
            if not row["asset_id_0"]:
                continue
            markets.append({
                "slug": row["slug"],
                "event_slug": row["event_slug"],
                "question": row["question"],
                "type": row["type"],
                "asset_id_yes": row["asset_id_0"],
                "condition_id": row["market_id"],
                "date": row["date"],
            })
        if markets:
            games[gds] = markets

    print(f"[discover] {len(games):,} games after filters "
          f"(full_set_only={full_set_only})", file=sys.stderr)
    return games


# ---------------------------------------------------------------------------
# Kickoff resolution
# ---------------------------------------------------------------------------

def _parse_iso(s):
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00").replace(" ", "T")
        # Strip anything past "+00:00" gracefully
        dt = datetime.fromisoformat(s.rstrip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _load_games_csv_kickoffs(csv_path):
    """Return {event_slug: kickoff_epoch} from one games.csv file."""
    out = {}
    if not csv_path.exists():
        return out
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ev = row.get("event_slug", "")
            gst = row.get("game_start_time", "")
            ts = _parse_iso(gst)
            if ev and ts is not None:
                out[ev] = ts
    return out


def _load_kickoff_memo():
    if KICKOFF_CACHE_PATH.exists():
        with open(KICKOFF_CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_kickoff_memo(memo):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(KICKOFF_CACHE_PATH, "w") as f:
        json.dump(memo, f, indent=2, sort_keys=True)


async def _gamma_lookup(client, event_slug_no_date, sem):
    """Fetch Gamma event by (un-dated) slug; return kickoff epoch or None.

    Telonex event_slug examples: `arg-aae-riv-2026-03-22`, `arg-aae-riv-2026-03-22-more-markets`.
    The Gamma event slug matches telonex event_slug exactly, so we use it directly.
    """
    async with sem:
        try:
            r = await client.get(
                f"{GAMMA_API}/events", params={"slug": event_slug_no_date}, timeout=15
            )
            if r.status_code != 200:
                return None
            data = r.json()
            if not data:
                return None
            ev = data[0] if isinstance(data, list) else data
            for m in ev.get("markets", []):
                ts = _parse_iso(m.get("gameStartTime", "")) or _parse_iso(ev.get("startDate", ""))
                if ts:
                    return ts
            return _parse_iso(ev.get("startDate", ""))
        except Exception:
            return None


async def resolve_kickoffs(games, skip_gamma=False, gamma_parallel=15):
    """Return {game_date_slug: kickoff_epoch}. Writes memo + skip list."""
    print("[kickoff] Loading cache files...", file=sys.stderr)

    # Memoised results from prior runs
    memo = _load_kickoff_memo()
    hits_memo = sum(1 for g in games if g in memo)

    # self_collected indexed by event_slug
    self_collected_map = _load_games_csv_kickoffs(SELF_COLLECTED_GAMES_CSV)
    print(f"[kickoff]   memo={len(memo):,} "
          f"self_collected={len(self_collected_map):,}", file=sys.stderr)

    resolved = dict(memo)
    unresolved = []

    for gds, markets in games.items():
        if gds in resolved:
            continue
        ts = None

        # Try every event_slug the game uses (main event + -more-markets)
        ev_slugs_used = {m["event_slug"] for m in markets}
        for ev in ev_slugs_used:
            ts = self_collected_map.get(ev)
            if ts:
                break

        if ts is not None:
            resolved[gds] = ts
        else:
            unresolved.append(gds)

    print(f"[kickoff]   resolved from caches: "
          f"{len(resolved) - hits_memo:,} new, {hits_memo:,} from memo",
          file=sys.stderr)

    # Gamma fallback for remaining
    if unresolved and not skip_gamma:
        print(f"[kickoff]   fetching {len(unresolved):,} from Gamma "
              f"(parallel={gamma_parallel})...", file=sys.stderr, flush=True)
        sem = asyncio.Semaphore(gamma_parallel)
        async with httpx.AsyncClient() as client:
            tasks = []
            for gds in unresolved:
                # Try the date-suffixed main-event slug first (e.g. arg-aae-riv-2026-03-22)
                tasks.append(_gamma_lookup(client, gds, sem))
            results = await asyncio.gather(*tasks)
        still_unresolved = []
        for gds, ts in zip(unresolved, results):
            if ts is not None:
                resolved[gds] = ts
            else:
                still_unresolved.append(gds)
        print(f"[kickoff]   Gamma resolved {len(unresolved)-len(still_unresolved):,}; "
              f"{len(still_unresolved):,} unresolved", file=sys.stderr)
        unresolved = still_unresolved

    _save_kickoff_memo(resolved)
    if unresolved:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(SKIPPED_PATH, "w") as f:
            for gds in sorted(unresolved):
                f.write(gds + "\n")
        print(f"[kickoff]   wrote {len(unresolved)} skips → {SKIPPED_PATH}",
              file=sys.stderr)

    return {g: resolved[g] for g in games if g in resolved}


# ---------------------------------------------------------------------------
# market_id int32 assignment
# ---------------------------------------------------------------------------

_market_id_cache = {}  # condition_id -> int32


def _load_market_id_map():
    """Load {condition_id: numeric_market_id} from the persisted cache."""
    if not MARKET_ID_CACHE_PATH.exists():
        return {}
    with open(MARKET_ID_CACHE_PATH) as f:
        return {k: int(v) for k, v in json.load(f).items()}


def _save_market_id_map(m):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(MARKET_ID_CACHE_PATH, "w") as f:
        json.dump({k: int(v) for k, v in m.items()}, f, indent=2, sort_keys=True)


async def _gamma_lookup_market_id(client, slug, sem):
    """Resolve a Polymarket market slug to Gamma's numeric market_id.

    Per docs/polymarket_market_metadata.md section 2d, the `slug` filter on
    /markets requires `closed` to match the market's state, so we try both.
    """
    async with sem:
        for closed in ("false", "true"):
            try:
                r = await client.get(
                    f"{GAMMA_API}/markets",
                    params={"slug": slug, "closed": closed},
                    timeout=15,
                )
                if r.status_code != 200:
                    continue
                data = r.json()
                if not data:
                    continue
                m = data[0] if isinstance(data, list) else data
                mid = m.get("id")
                if mid:
                    return int(mid)
            except Exception:
                continue
        return None


async def resolve_market_ids(games, market_id_map, gamma_parallel=15):
    """Ensure every market in `games` has a numeric Gamma market_id.

    Fetches missing ids via Gamma's /markets?slug={slug} endpoint and
    persists them. Merges into the existing cache so previously-resolved
    entries are never lost.
    """
    needed = {}  # condition_id -> slug
    for markets in games.values():
        for m in markets:
            cid = m["condition_id"]
            if cid in market_id_map:
                continue
            needed[cid] = m["slug"]

    if not needed:
        return market_id_map

    print(f"[market_id] fetching {len(needed):,} numeric ids from Gamma "
          f"(parallel={gamma_parallel})...", file=sys.stderr, flush=True)
    sem = asyncio.Semaphore(gamma_parallel)
    async with httpx.AsyncClient() as client:
        cids = list(needed.keys())
        tasks = [_gamma_lookup_market_id(client, needed[c], sem) for c in cids]
        results = await asyncio.gather(*tasks)

    n_ok = n_fail = 0
    for cid, mid in zip(cids, results):
        if mid is not None:
            market_id_map[cid] = mid
            n_ok += 1
        else:
            n_fail += 1
    print(f"[market_id]   resolved {n_ok:,}; failed {n_fail:,}", file=sys.stderr)

    _save_market_id_map(market_id_map)
    return market_id_map


def resolve_market_id(condition_id, cache_map):
    """int32 market_id for self_collected-compatible schema.

    Expects the condition_id to be present in cache_map (populated by
    resolve_market_ids before any per_game data is written). If Gamma
    resolution failed for this id, falls back to crc32 with a warning —
    the per-game parquet will still write, but with a synthetic id.
    """
    if condition_id in cache_map:
        return cache_map[condition_id]
    print(f"[market_id] WARNING: no Gamma id for {condition_id}; "
          f"using crc32 fallback", file=sys.stderr)
    return zlib.crc32(condition_id.encode()) & 0x7fffffff


# ---------------------------------------------------------------------------
# Per-market download + extract
# ---------------------------------------------------------------------------

def _extract_window_frame(raw_path, market_info, kickoff_ts, window_hours, market_id_int):
    """Load a downloaded quotes parquet, clip to [last-pre-window update,
    window-end], cast, emit YES+NO rows."""
    # Window in microseconds
    pre_s = 0.5 * 3600
    post_s = (window_hours - 0.5) * 3600
    t_lo_us = int((kickoff_ts - pre_s) * 1_000_000)
    t_hi_us = int((kickoff_ts + post_s) * 1_000_000)

    df = pq.read_table(raw_path, columns=[
        "timestamp_us", "bid_price", "bid_size", "ask_price", "ask_size"
    ]).to_pandas()

    # Seed window state with the most recent pre-window update (if any), so
    # downstream ffill has a value to carry across the window start instead
    # of leaving the market on the sentinel until its first in-window quote.
    ts = df["timestamp_us"]
    pre = ts[ts < t_lo_us]
    t_seed_us = int(pre.max()) if not pre.empty else t_lo_us
    df = df[(ts >= t_seed_us) & (ts <= t_hi_us)]
    if df.empty:
        return None

    # Cast price/size strings to float32 (None → NaN)
    for c in ("bid_price", "ask_price", "bid_size", "ask_size"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    # Translate NaN prices to one-sided-book sentinels so the synthesis
    # downstream stays consistent. NaN here means "no liquidity on that side"
    # (typically late-game when one outcome is near-certain). If we left NaN
    # in place, pivot_table(aggfunc='last') would skip the cell, ffill would
    # carry forward a stale prior value on that side while the complementary
    # side updated freshly, producing fake "crossed-book" rows downstream.
    nan_bid = df["bid_price"].isna()
    nan_ask = df["ask_price"].isna()
    df.loc[nan_bid, "bid_price"] = 0.0
    df.loc[nan_bid, "bid_size"] = 0.0
    df.loc[nan_ask, "ask_price"] = 1.0
    df.loc[nan_ask, "ask_size"] = 0.0

    ts_ms = (df["timestamp_us"].to_numpy() // 1000).astype(np.int64)
    yes_bid = df["bid_price"].to_numpy()
    yes_ask = df["ask_price"].to_numpy()
    yes_bid_size = df["bid_size"].to_numpy()
    yes_ask_size = df["ask_size"].to_numpy()
    n = len(df)

    market_type = market_info["type"]
    # Map telonex/Gamma canonical → self_collected canonical ("spreads" → "spread")
    type_out = {"spreads": "spread", "both_teams_to_score": "btts"}.get(market_type, market_type)

    common = {
        "market_id": np.full(n * 2, market_id_int, dtype=np.int32),
        "condition_id": [market_info["condition_id"]] * (n * 2),
        "type": [type_out] * (n * 2),
        "question": [market_info["question"]] * (n * 2),
        "timestamp_ms": np.concatenate([ts_ms, ts_ms]),
        "is_yes": np.concatenate([
            np.ones(n, dtype=bool), np.zeros(n, dtype=bool)
        ]),
        "best_bid": np.concatenate([yes_bid, (1.0 - yes_ask).astype(np.float32)]),
        "best_ask": np.concatenate([yes_ask, (1.0 - yes_bid).astype(np.float32)]),
        "best_bid_size": np.concatenate([yes_bid_size, yes_ask_size]),
        "best_ask_size": np.concatenate([yes_ask_size, yes_bid_size]),
    }
    return pd.DataFrame(common)


async def _download_market(client, api_key, sem, market_info, kickoff_ts,
                           window_hours, market_id_int, tmpdir):
    """Download one market's quotes file, extract windowed frame, delete raw."""
    asset = market_info["asset_id_yes"]
    date = market_info["date"]
    raw_path = Path(tmpdir) / f"{asset[:16]}_{date}.parquet"
    try:
        size = await download_quotes(client, api_key, date, asset, raw_path, sem=sem)
    except TelonexError as e:
        print(f"    SKIP {market_info['slug']}: {e}", file=sys.stderr)
        return None
    try:
        frame = _extract_window_frame(raw_path, market_info, kickoff_ts,
                                      window_hours, market_id_int)
    finally:
        try:
            raw_path.unlink()
        except FileNotFoundError:
            pass
    return frame


# ---------------------------------------------------------------------------
# Per-game processing
# ---------------------------------------------------------------------------

async def process_game(client, api_key, sem, gds, markets, kickoff_ts,
                       window_hours, market_id_map, tmpdir, force):
    out_path = SNAPSHOTS_DIR / f"{gds}.parquet"
    if out_path.exists() and not force:
        return "skip-exists", 0

    tasks = []
    for m in markets:
        mid = resolve_market_id(m["condition_id"], market_id_map)
        tasks.append(_download_market(client, api_key, sem, m, kickoff_ts,
                                      window_hours, mid, tmpdir))
    frames = await asyncio.gather(*tasks)
    frames = [f for f in frames if f is not None]
    if not frames:
        return "no-data", 0

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("timestamp_ms", kind="stable", inplace=True, ignore_index=True)

    table = pa.Table.from_pandas(combined, preserve_index=False, schema=pa.schema([
        ("market_id", pa.int32()),
        ("condition_id", pa.string()),
        ("type", pa.string()),
        ("question", pa.string()),
        ("is_yes", pa.bool_()),
        ("timestamp_ms", pa.int64()),
        ("best_bid", pa.float32()),
        ("best_ask", pa.float32()),
        ("best_bid_size", pa.float32()),
        ("best_ask_size", pa.float32()),
    ]))
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")
    return "ok", len(combined)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args):
    load_dotenv(Path(project_root) / ".env")
    api_key = os.getenv("TELONEX_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: TELONEX_API_KEY not set in .env", file=sys.stderr)
        return 2

    games_filter = set(args.games.split(",")) if args.games else None
    league_filter = set(args.league.split(",")) if args.league else None

    games = discover_games(
        METADATA_PATH,
        start_date=args.start,
        end_date=args.end,
        games_filter=games_filter,
        league_filter=league_filter,
        full_set_only=not args.no_full_set,
    )
    if not games:
        print("[main] No games match the filters.", file=sys.stderr)
        return 0

    # Kickoff lookup
    kickoffs = await resolve_kickoffs(games, skip_gamma=args.dry_run)
    games = {g: markets for g, markets in games.items() if g in kickoffs}
    total_markets = sum(len(v) for v in games.values())
    print(f"[main] {len(games):,} games with kickoff resolved; "
          f"{total_markets:,} markets total", file=sys.stderr)

    if args.dry_run:
        # Estimate size: 4h quotes ≈ 0.23 MB / active market + ~0.03 MB / btts
        est_bytes = 0
        for markets in games.values():
            for m in markets:
                est_bytes += 30_000 if m["type"] == "both_teams_to_score" else 230_000
        print(f"[dry-run] downloads={total_markets:,} "
              f"est={est_bytes/1e6:.1f} MB raw (before windowing)",
              file=sys.stderr)
        print(f"[dry-run] output: {len(games)} parquets in {SNAPSHOTS_DIR}",
              file=sys.stderr)
        return 0

    market_id_map = _load_market_id_map()
    print(f"[main] market_id cache: {len(market_id_map):,} condition_ids",
          file=sys.stderr)
    market_id_map = await resolve_market_ids(games, market_id_map)

    sem = asyncio.Semaphore(args.parallel)
    ok = err = skipped = 0
    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix="telonex_") as tmpdir:
        # httpx: long timeout for R2 redirect + payload; follow 302 to signed URL
        timeout = httpx.Timeout(60.0, connect=10.0)
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            # Process games sequentially; each game's ≤12 downloads run concurrently
            for i, (gds, markets) in enumerate(sorted(games.items()), 1):
                kickoff = kickoffs[gds]
                try:
                    status, n = await process_game(
                        client, api_key, sem, gds, markets, kickoff,
                        args.window_hours, market_id_map, tmpdir, args.force,
                    )
                except Exception as e:
                    print(f"  [{i}/{len(games)}] {gds}: ERROR {e!r}", file=sys.stderr)
                    err += 1
                    continue

                if status == "ok":
                    ok += 1
                    print(f"  [{i}/{len(games)}] {gds}: wrote {n:,} rows",
                          file=sys.stderr, flush=True)
                elif status == "skip-exists":
                    skipped += 1
                else:
                    err += 1
                    print(f"  [{i}/{len(games)}] {gds}: {status}", file=sys.stderr)

    print(f"[main] done: ok={ok:,} skipped={skipped:,} err={err:,} "
          f"elapsed={time.time()-t0:.0f}s", file=sys.stderr)
    return 0 if err == 0 else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", help="earliest game date (YYYY-MM-DD)", default="2025-10-11")
    p.add_argument("--end", help="latest game date (YYYY-MM-DD)", default=None)
    p.add_argument("--games", help="comma-separated game_date_slugs", default=None)
    p.add_argument("--league", help="comma-separated tag_slugs (e.g. epl,bun)", default=None)
    p.add_argument("--no-full-set", action="store_true",
                   help="include games missing some of the 3+4+4+1 canonical markets")
    p.add_argument("--window-hours", type=float, default=4.0,
                   help="consolidation window (0.5h pre + (window-0.5)h post kickoff)")
    p.add_argument("--parallel", type=int, default=10,
                   help="concurrent downloads (shared semaphore)")
    p.add_argument("--force", action="store_true",
                   help="re-process games even if per_game_data already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="enumerate targets + estimate without downloading")
    args = p.parse_args()

    if args.end is None:
        args.end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
