#!/usr/bin/env python3
"""Build soccer dataset from self_collected WS logs, analogous to the Dome dataset.

Scans all self_collected WS log files, discovers games via Gamma API enrichment,
then extracts best bid/ask time series per game in the [-30min, +180min]
window around game start.

Outputs:
    data/self_collected/games.csv                     — same schema as data/dome/games.csv
    data/self_collected/per_game_data/{game_slug}.parquet  — same schema as data/dome/per_game_data/

Usage:
    python -m pipelines.self_collected.build_dataset
    python -m pipelines.self_collected.build_dataset --games bun-aug-hof
    python -m pipelines.self_collected.build_dataset --self_collected-dir data/raw_ws_logs
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import httpx
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pregame_pca.pipelines.filtering import filter_self_collected
from pregame_pca.polymarket.sdk import MarketTracker
from pregame_pca.polymarket.soccer import SPORTS_TYPE_MAP, parse_game_slug as _parse_game_slug

# self_collected side encoding → WS format (matches scripts/backtest_arb_monitor_self_collected.py:48)
_SIDE_STR = {0: "BUY", 1: "SELL"}

GAMMA_API = "https://gamma-api.polymarket.com"
_DATA_DIR = Path(project_root) / "data" / "self_collected"
SNAPSHOTS_DIR = _DATA_DIR / "per_game_data"

PRE_GAME_MIN = 30
POST_GAME_MIN = 180

CSV_FIELDS = [
    "game_slug", "game_id", "game_start_time", "event_slug",
    "market_id", "condition_id", "type", "question",
    "yes_token", "no_token",
]

MIN_EVENTS = 1000  # skip self_collected files with fewer events


# ---------------------------------------------------------------------------
# Phase 1: Scan self_collected files and discover games
# ---------------------------------------------------------------------------

def scan_self_collected_files(self_collected_dir):
    """Scan all soccer self_collected files.

    Returns:
        self_collected_index: list of {
            path, meta_markets (list of str), t_min_ms, t_max_ms
        }
        all_market_ids: set of all unique Gamma market IDs across files
    """
    files = sorted(Path(self_collected_dir).glob("ws_log_soccer_*.h5"))
    print(f"[scan] Found {len(files)} self_collected files", file=sys.stderr, flush=True)

    self_collected_index = []
    all_market_ids = set()

    for fpath in files:
        try:
            f = h5py.File(fpath, "r")
        except OSError as e:
            print(f"[scan]   SKIP {fpath.name}: {e}", file=sys.stderr, flush=True)
            continue

        try:
            meta = [m.decode() if isinstance(m, bytes) else m
                    for m in f["/meta/markets"][:]]
            pc = f["/market/price_change"]
            n = pc["timestamp"].shape[0]

            if n < MIN_EVENTS:
                print(f"[scan]   SKIP {fpath.name}: only {n} events",
                      file=sys.stderr, flush=True)
                f.close()
                continue

            # Skip files with empty market IDs
            if not meta[0]:
                print(f"[scan]   SKIP {fpath.name}: empty market IDs",
                      file=sys.stderr, flush=True)
                f.close()
                continue

            t_min = int(pc["timestamp"][0])
            t_max = int(pc["timestamp"][-1])
            f.close()

            self_collected_index.append({
                "path": str(fpath),
                "meta_markets": meta,
                "t_min_ms": t_min,
                "t_max_ms": t_max,
            })
            all_market_ids.update(meta)

            dt0 = datetime.fromtimestamp(t_min / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            dt1 = datetime.fromtimestamp(t_max / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            print(f"[scan]   {fpath.name}: {len(meta)} mkts, {n:,} events, "
                  f"{dt0} → {dt1}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[scan]   SKIP {fpath.name}: {e}", file=sys.stderr, flush=True)
            f.close()
            continue

    print(f"[scan] {len(self_collected_index)} usable files, {len(all_market_ids)} unique market IDs",
          file=sys.stderr, flush=True)
    return self_collected_index, all_market_ids


async def enrich_from_gamma(market_ids, parallel=15):
    """Fetch metadata from Gamma /markets/{id} for each market ID.

    Returns dict: market_id_str -> {
        condition_id, game_start_time, question, type,
        yes_token, no_token, game_slug, event_slug, game_id
    }
    """
    ids = sorted(market_ids)
    print(f"[enrich] Fetching {len(ids)} markets from Gamma...",
          file=sys.stderr, flush=True)

    sem = asyncio.Semaphore(parallel)
    results = {}
    t0 = time.time()

    async def fetch_one(client, mid):
        async with sem:
            for attempt in range(4):
                try:
                    r = await client.get(f"{GAMMA_API}/markets/{mid}")
                    if r.status_code == 200:
                        return mid, r.json()
                    if r.status_code == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                except Exception:
                    if attempt < 3:
                        await asyncio.sleep(1)
        return mid, None

    async with httpx.AsyncClient(timeout=30) as client:
        tasks = [fetch_one(client, mid) for mid in ids]
        batch_size = 500
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch)
            for mid, data in batch_results:
                if data is None:
                    continue

                smt = data.get("sportsMarketType", "")
                market_type = SPORTS_TYPE_MAP.get(smt)
                if not market_type:
                    continue

                raw_ids = data.get("clobTokenIds")
                if isinstance(raw_ids, str):
                    raw_ids = json.loads(raw_ids)
                if not raw_ids or len(raw_ids) < 2:
                    continue

                slug = data.get("slug", "")
                game_slug = _parse_game_slug(slug)

                results[mid] = {
                    "condition_id": data.get("conditionId", ""),
                    "game_start_time": data.get("gameStartTime", ""),
                    "question": data.get("question", ""),
                    "type": market_type,
                    "yes_token": raw_ids[0],
                    "no_token": raw_ids[1],
                    "game_slug": game_slug,
                    "event_slug": slug,
                    "game_id": data.get("gameId"),
                }

            elapsed = time.time() - t0
            done = min(i + batch_size, len(tasks))
            print(f"[enrich] {done}/{len(tasks)} ({elapsed:.0f}s)",
                  file=sys.stderr, flush=True)

    # Propagate game_id
    slug_to_gid = {}
    for info in results.values():
        if info["game_id"]:
            slug_to_gid[info["game_slug"]] = info["game_id"]
    for info in results.values():
        if not info["game_id"]:
            info["game_id"] = slug_to_gid.get(info["game_slug"])

    print(f"[enrich] Got metadata for {len(results)} markets",
          file=sys.stderr, flush=True)
    return results


def build_games_csv(market_meta, out_path):
    """Write games.csv from enriched market metadata."""
    rows = []
    for mid, info in sorted(market_meta.items(), key=lambda x: (x[1]["game_slug"], x[1]["type"], x[0])):
        rows.append({
            "game_slug": info["game_slug"],
            "game_id": info["game_id"] or "",
            "game_start_time": info["game_start_time"],
            "event_slug": info["event_slug"],
            "market_id": mid,
            "condition_id": info["condition_id"],
            "type": info["type"],
            "question": info["question"],
            "yes_token": info["yes_token"],
            "no_token": info["no_token"],
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    games = set(r["game_slug"] for r in rows)
    print(f"[csv] Wrote {len(rows)} markets across {len(games)} games → {out_path}",
          file=sys.stderr, flush=True)
    return rows


# ---------------------------------------------------------------------------
# Phase 2: Extract per-game parquets from self_collected events
# ---------------------------------------------------------------------------

def _build_book_id_spans(bl_book_id):
    """Return (sort_by_book_id_indices, {book_id: (start, end)}).

    Copied from scripts/backtest_arb_monitor_self_collected.py:_build_book_id_spans so
    we can slice book_levels rows by book_id without a full O(n) scan per
    lookup.
    """
    sort_by_bid = np.argsort(bl_book_id, kind="stable")
    sorted_bids = bl_book_id[sort_by_bid]
    change_pts = np.where(np.diff(sorted_bids) != 0)[0] + 1
    starts = np.concatenate([[0], change_pts])
    ends = np.concatenate([change_pts, [len(sorted_bids)]])
    spans = {int(sorted_bids[s]): (int(s), int(e)) for s, e in zip(starts, ends)}
    return sort_by_bid, spans


def _build_book_snapshot(j, bm_book_id, bl_price, bl_side, bl_size,
                          sort_by_bid, book_id_spans):
    """Assemble (asks, bids) lists from a single bm event index. Returns
    (asks, bids) or None if the book_id span is missing."""
    book_id_val = int(bm_book_id[j])
    span = book_id_spans.get(book_id_val)
    if span is None:
        return None
    s_lvl, e_lvl = span
    li_range = sort_by_bid[s_lvl:e_lvl]
    asks = []
    bids = []
    for li in li_range:
        lvl = {"price": float(bl_price[li]), "size": float(bl_size[li])}
        if bl_side[li] == 1:
            asks.append(lvl)
        else:
            bids.append(lvl)
    return asks, bids


def _apply_pre_window_book_seeds(tracker, file_events, self_collected_index, win_start,
                                  market_ids_set, mid_to_yes_token,
                                  mid_to_no_token, needed_keys):
    """Seed the MarketTracker with the most-recent book_meta per_game_data per
    (mid, outcome) strictly before win_start. Without this, price_change
    events at the start of the replay window land on an empty book and the
    tracker reports the (best_ask=1.0, best_bid=0.0) sentinel — see
    src/polymarket/sdk.py:599-636. Returns (n_seeded, n_needed).

    Phase 1 reuses arrays already loaded for covering files. Phase 2 opens
    earlier files in reverse-time order until every needed token is seeded
    or the index is exhausted.
    """
    seeds = {}  # key=(mid_int, outcome_str) -> (ts, snapshot_dict)

    def _record_candidate(j, bm_ts, bm_midx, bm_iy, bm_book_id,
                          bl_price, bl_side, bl_size, sort_by_bid, book_id_spans,
                          idx_to_mid):
        midx = int(bm_midx[j])
        if midx not in idx_to_mid:
            return
        iy = bool(bm_iy[j])
        mid_int = idx_to_mid[midx]
        outcome = "YES" if iy else "NO"
        key = (mid_int, outcome)
        if key not in needed_keys:
            return
        ts = int(bm_ts[j])
        prev = seeds.get(key)
        if prev is not None and prev[0] >= ts:
            return
        snap = _build_book_snapshot(j, bm_book_id, bl_price, bl_side, bl_size,
                                    sort_by_bid, book_id_spans)
        if snap is None:
            return
        asks, bids = snap
        token_id = mid_to_yes_token[mid_int] if iy else mid_to_no_token[mid_int]
        seeds[key] = (ts, {
            "asset_id": token_id, "timestamp": ts, "asks": asks, "bids": bids,
        })

    # Phase 1: scan covering files that span win_start (arrays already loaded).
    for fe in file_events:
        idx_to_mid = fe["idx_to_mid"]
        bm_ts = fe["bm_ts"]; bm_midx = fe["bm_midx"]
        if bm_ts.size == 0:
            continue
        valid_idxs = np.array(list(idx_to_mid.keys()), dtype=np.uint16)
        keep = (bm_ts < win_start) & np.isin(bm_midx, valid_idxs)
        if not keep.any():
            continue
        # For each (midx, iy) take only the latest pre-window bm event.
        pre_idx = np.nonzero(keep)[0]
        keys_arr = bm_midx[pre_idx].astype(np.int64) * 2 + fe["bm_iy"][pre_idx].astype(np.int64)
        order = np.lexsort((bm_ts[pre_idx], keys_arr))
        keys_sorted = keys_arr[order]
        if keys_sorted.size == 0:
            continue
        change = np.concatenate([[True], keys_sorted[1:] != keys_sorted[:-1]])
        seg_start = np.nonzero(change)[0]
        seg_end = np.concatenate([seg_start[1:], [keys_sorted.size]])
        last_in_seg = seg_end - 1
        for s_rel in order[last_in_seg]:
            j = int(pre_idx[s_rel])
            _record_candidate(
                j, bm_ts, bm_midx, fe["bm_iy"], fe["bm_book_id"],
                fe["bl_price"], fe["bl_side"], fe["bl_size"],
                fe["sort_by_bid"], fe["book_id_spans"], idx_to_mid,
            )

    # Phase 2: open earlier files (t_max_ms < win_start) for tokens still missing.
    still_missing = needed_keys - set(seeds.keys())
    if still_missing:
        relevant_mid_strs = {str(mid) for mid, _ in still_missing}
        earlier = [
            e for e in self_collected_index
            if e["t_max_ms"] < win_start
            and (set(e["meta_markets"]) & relevant_mid_strs)
        ]
        earlier.sort(key=lambda e: e["t_max_ms"], reverse=True)
        for entry in earlier:
            if not still_missing:
                break
            try:
                f = h5py.File(entry["path"], "r")
            except OSError:
                continue
            try:
                meta = entry["meta_markets"]
                idx_to_mid_e = {i: int(m) for i, m in enumerate(meta) if m in market_ids_set}
                file_mids = set(idx_to_mid_e.values())
                if not any(mid in file_mids for mid, _ in still_missing):
                    continue
                bm = f["/market/book_meta"]
                bm_ts = bm["timestamp"][:].astype(np.int64)
                bm_midx = bm["market_idx"][:].astype(np.uint16)
                bm_iy = bm["is_yes"][:].astype(bool)
                bm_book_id = bm["book_id"][:]
                bl = f["/market/book_levels"]
                bl_book_id = bl["book_id"][:]
                bl_price = bl["price"][:]
                bl_side = bl["side"][:].astype(np.uint8)
                bl_size = bl["size"][:]
            finally:
                f.close()
            sort_by_bid_e, book_id_spans_e = _build_book_id_spans(bl_book_id)
            valid_idxs_e = np.array(list(idx_to_mid_e.keys()), dtype=np.uint16)
            keep = (bm_ts < win_start) & np.isin(bm_midx, valid_idxs_e)
            if not keep.any():
                continue
            indices = np.nonzero(keep)[0]
            for ord_idx in np.argsort(bm_ts[indices])[::-1]:
                if not still_missing:
                    break
                j = int(indices[ord_idx])
                midx = int(bm_midx[j])
                iy = bool(bm_iy[j])
                if midx not in idx_to_mid_e:
                    continue
                mid_int = idx_to_mid_e[midx]
                outcome = "YES" if iy else "NO"
                key = (mid_int, outcome)
                if key not in still_missing:
                    continue
                _record_candidate(
                    j, bm_ts, bm_midx, bm_iy, bm_book_id,
                    bl_price, bl_side, bl_size, sort_by_bid_e, book_id_spans_e,
                    idx_to_mid_e,
                )
                still_missing.discard(key)

    # Apply seeds in chronological order (cosmetic — per-token they are
    # independent, but a deterministic order makes debug logs reproducible).
    for ts, snap in sorted(seeds.values(), key=lambda x: x[0]):
        tracker.process_book(snap)

    return len(seeds), len(needed_keys)


def find_covering_files(game_start_ms, self_collected_index, market_ids_set):
    """Find self_collected files that cover the game window AND contain at least one market."""
    win_start = game_start_ms - PRE_GAME_MIN * 60_000
    win_end = game_start_ms + POST_GAME_MIN * 60_000
    covering = []
    for entry in self_collected_index:
        # Time overlap?
        if entry["t_max_ms"] < win_start or entry["t_min_ms"] > win_end:
            continue
        # Has any of the game's markets?
        self_collected_mids = set(entry["meta_markets"])
        if not self_collected_mids & market_ids_set:
            continue
        covering.append(entry)
    return covering, win_start, win_end


def extract_game_events(game_slug, game_markets, self_collected_index, market_meta):
    """Extract per-event order-book state for one game via MarketTracker replay.

    For each price_change event in the game's time window, emit a row carrying
    best_bid / best_ask / best_bid_size / best_ask_size as of that moment.
    book_meta per_game_data are applied to the tracker but do NOT emit rows — row
    emission semantics match the pre-replay pipeline (one row per price_change
    event per (market, side)), so the only schema change is the two new size
    columns. Replay uses the same pattern as scripts/backtest_arb_monitor_self_collected.py.

    game_markets: list of market_id strings for this game
    Returns a DataFrame with the new schema, or None.
    """
    # Get game_start_time (unchanged from original)
    gst_str = None
    for mid in game_markets:
        gst_str = market_meta[mid].get("game_start_time")
        if gst_str:
            break
    if not gst_str:
        return None

    try:
        dt = datetime.fromisoformat(
            gst_str.replace("Z", "+00:00").replace("+00 ", "+00:00 ").rstrip()
        )
        game_start_ms = int(dt.timestamp() * 1000)
    except Exception:
        return None

    market_ids_set = set(game_markets)
    covering, win_start, win_end = find_covering_files(game_start_ms, self_collected_index, market_ids_set)
    if not covering:
        return None

    # Build a MarketTracker instance for just this game's tokens.
    token_to_key = {}
    mid_to_yes_token = {}
    mid_to_no_token = {}
    for mid in game_markets:
        info = market_meta[mid]
        yes_tok = info["yes_token"]
        no_tok = info["no_token"]
        mid_int = int(mid)
        token_to_key[yes_tok] = (mid_int, "YES")
        token_to_key[no_tok] = (mid_int, "NO")
        mid_to_yes_token[mid_int] = yes_tok
        mid_to_no_token[mid_int] = no_tok
    tracker = MarketTracker(token_to_key, list(token_to_key.keys()))

    # Collect per-file event arrays so we can merge into one chronological stream.
    file_events = []
    for entry in covering:
        try:
            f = h5py.File(entry["path"], "r")
        except OSError:
            continue
        meta = entry["meta_markets"]

        idx_to_mid = {}
        for i, mid in enumerate(meta):
            if mid in market_ids_set:
                idx_to_mid[i] = int(mid)
        if not idx_to_mid:
            f.close()
            continue
        valid_idxs = np.array(list(idx_to_mid.keys()), dtype=np.uint16)

        try:
            pc = f["/market/price_change"]
            pc_ts = pc["timestamp"][:].astype(np.int64)
            pc_midx = pc["market_idx"][:].astype(np.uint16)
            pc_iy = pc["is_yes"][:].astype(bool)
            pc_price = pc["price"][:]
            pc_size = pc["size"][:]
            pc_side = pc["side"][:].astype(np.uint8)

            bm = f["/market/book_meta"]
            bm_ts = bm["timestamp"][:].astype(np.int64)
            bm_midx = bm["market_idx"][:].astype(np.uint16)
            bm_iy = bm["is_yes"][:].astype(bool)
            bm_book_id = bm["book_id"][:]

            bl = f["/market/book_levels"]
            bl_book_id = bl["book_id"][:]
            bl_price = bl["price"][:]
            bl_side = bl["side"][:].astype(np.uint8)
            bl_size = bl["size"][:]
        finally:
            f.close()

        pc_keep = (pc_ts >= win_start) & (pc_ts <= win_end) & np.isin(pc_midx, valid_idxs)
        bm_keep = (bm_ts >= win_start) & (bm_ts <= win_end) & np.isin(bm_midx, valid_idxs)
        pc_idx = np.nonzero(pc_keep)[0]
        bm_idx = np.nonzero(bm_keep)[0]
        if pc_idx.size == 0 and bm_idx.size == 0:
            continue

        sort_by_bid, book_id_spans = _build_book_id_spans(bl_book_id)
        file_events.append({
            "idx_to_mid": idx_to_mid,
            "pc_idx": pc_idx,
            "pc_ts": pc_ts, "pc_midx": pc_midx, "pc_iy": pc_iy,
            "pc_price": pc_price, "pc_size": pc_size, "pc_side": pc_side,
            "bm_idx": bm_idx,
            "bm_ts": bm_ts, "bm_midx": bm_midx, "bm_iy": bm_iy,
            "bm_book_id": bm_book_id,
            "bl_price": bl_price, "bl_side": bl_side, "bl_size": bl_size,
            "sort_by_bid": sort_by_bid,
            "book_id_spans": book_id_spans,
        })

    if not file_events:
        return None

    # Seed the tracker with the latest pre-window book_meta per token so that
    # in-window price_change events apply to a real book state instead of the
    # empty-book sentinel.
    needed_keys = set()
    for mid in game_markets:
        mid_int = int(mid)
        needed_keys.add((mid_int, "YES"))
        needed_keys.add((mid_int, "NO"))
    _apply_pre_window_book_seeds(
        tracker, file_events, self_collected_index, win_start, market_ids_set,
        mid_to_yes_token, mid_to_no_token, needed_keys,
    )

    # Merge events from all covering files into one chronological stream.
    merged_ts_parts = []
    merged_kind_parts = []
    merged_file_parts = []
    merged_orig_parts = []
    for fi, fe in enumerate(file_events):
        if fe["pc_idx"].size > 0:
            merged_ts_parts.append(fe["pc_ts"][fe["pc_idx"]])
            merged_kind_parts.append(np.zeros(fe["pc_idx"].size, dtype=np.uint8))
            merged_file_parts.append(np.full(fe["pc_idx"].size, fi, dtype=np.uint16))
            merged_orig_parts.append(fe["pc_idx"])
        if fe["bm_idx"].size > 0:
            merged_ts_parts.append(fe["bm_ts"][fe["bm_idx"]])
            merged_kind_parts.append(np.ones(fe["bm_idx"].size, dtype=np.uint8))
            merged_file_parts.append(np.full(fe["bm_idx"].size, fi, dtype=np.uint16))
            merged_orig_parts.append(fe["bm_idx"])

    if not merged_ts_parts:
        return None

    merged_ts = np.concatenate(merged_ts_parts)
    merged_kind = np.concatenate(merged_kind_parts)
    merged_file = np.concatenate(merged_file_parts)
    merged_orig = np.concatenate(merged_orig_parts)
    order = np.argsort(merged_ts, kind="stable")

    # Replay. Collect one output row per price_change event (book events update
    # the tracker but don't emit).
    n_pc = int((merged_kind == 0).sum())
    out_mid = np.empty(n_pc, dtype=np.int64)
    out_isy = np.empty(n_pc, dtype=bool)
    out_ts = np.empty(n_pc, dtype=np.int64)
    out_bb = np.empty(n_pc, dtype=np.float32)
    out_ba = np.empty(n_pc, dtype=np.float32)
    out_bbs = np.empty(n_pc, dtype=np.float32)
    out_bas = np.empty(n_pc, dtype=np.float32)
    out_cursor = 0

    for k in order:
        ts_ms = int(merged_ts[k])
        kind = merged_kind[k]
        fi = int(merged_file[k])
        orig = int(merged_orig[k])
        fe = file_events[fi]

        if kind == 0:
            # price_change event → apply, then per_game_data tracker state
            midx = int(fe["pc_midx"][orig])
            iy = bool(fe["pc_iy"][orig])
            mid_int = fe["idx_to_mid"][midx]
            outcome = "YES" if iy else "NO"
            token = mid_to_yes_token[mid_int] if iy else mid_to_no_token[mid_int]
            side_int = int(fe["pc_side"][orig])
            tracker.process_price_change({
                "timestamp": ts_ms,
                "price_changes": [{
                    "asset_id": token,
                    "price": float(fe["pc_price"][orig]),
                    "size": float(fe["pc_size"][orig]),
                    "side": _SIDE_STR[side_int],
                }],
            })
            bb_p, bb_s = tracker.get_best_bid_with_size(mid_int, outcome)
            ba_p, ba_s = tracker.get_best_ask_with_size(mid_int, outcome)
            out_mid[out_cursor] = mid_int
            out_isy[out_cursor] = iy
            out_ts[out_cursor] = ts_ms
            out_bb[out_cursor] = bb_p
            out_ba[out_cursor] = ba_p
            out_bbs[out_cursor] = bb_s
            out_bas[out_cursor] = ba_s
            out_cursor += 1
        else:
            # book_meta event → assemble the per_game_data and apply (no row emission)
            midx = int(fe["bm_midx"][orig])
            iy = bool(fe["bm_iy"][orig])
            mid_int = fe["idx_to_mid"][midx]
            token = mid_to_yes_token[mid_int] if iy else mid_to_no_token[mid_int]
            book_id_val = int(fe["bm_book_id"][orig])
            span = fe["book_id_spans"].get(book_id_val)
            if span is None:
                continue
            s_lvl, e_lvl = span
            li_range = fe["sort_by_bid"][s_lvl:e_lvl]
            asks = []
            bids = []
            for li in li_range:
                entry = {
                    "price": float(fe["bl_price"][li]),
                    "size": float(fe["bl_size"][li]),
                }
                if fe["bl_side"][li] == 1:
                    asks.append(entry)
                else:
                    bids.append(entry)
            tracker.process_book({
                "asset_id": token,
                "timestamp": ts_ms,
                "asks": asks,
                "bids": bids,
            })

    if out_cursor == 0:
        return None

    out_mid = out_mid[:out_cursor]
    out_isy = out_isy[:out_cursor]
    out_ts = out_ts[:out_cursor]
    out_bb = out_bb[:out_cursor]
    out_ba = out_ba[:out_cursor]
    out_bbs = out_bbs[:out_cursor]
    out_bas = out_bas[:out_cursor]

    # Attach per-row condition_id / type / question via market_id lookup
    unique_mids, inverse = np.unique(out_mid, return_inverse=True)
    cond_lookup = np.empty(len(unique_mids), dtype=object)
    type_lookup = np.empty(len(unique_mids), dtype=object)
    q_lookup = np.empty(len(unique_mids), dtype=object)
    for i, m in enumerate(unique_mids):
        info = market_meta[str(int(m))]
        cond_lookup[i] = info["condition_id"]
        type_lookup[i] = info["type"]
        q_lookup[i] = info["question"]

    df = pd.DataFrame({
        "market_id": out_mid.astype(np.int32),
        "condition_id": cond_lookup[inverse],
        "type": type_lookup[inverse],
        "question": q_lookup[inverse],
        "is_yes": out_isy,
        "timestamp_ms": out_ts,
        "best_bid": out_bb,
        "best_ask": out_ba,
        "best_bid_size": out_bbs,
        "best_ask_size": out_bas,
    })
    df.sort_values("timestamp_ms", inplace=True, kind="stable")
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Build soccer dataset from self_collected WS logs")
    parser.add_argument("--self_collected-dir", default="data/raw_ws_logs",
                        help="Directory containing ws_log_soccer_*.h5 files")
    parser.add_argument("--out-csv", default=str(_DATA_DIR / "games.csv"),
                        help="Output games CSV")
    parser.add_argument("--out-dir", default=str(SNAPSHOTS_DIR),
                        help="Output directory for per-game parquets")
    parser.add_argument("--games", default="",
                        help="Comma-separated game slugs (default: all)")
    parser.add_argument("--parallel", type=int, default=15,
                        help="Concurrent Gamma API calls")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if parquet exists")
    parser.add_argument("--no-filter", action="store_true",
                        help="Skip the post-build session-coverage filter")
    parser.add_argument("--filter-out-dir",
                        default=str(Path(project_root) / "data" / "comparisons"),
                        help="Directory for filter diagnostics")
    args = parser.parse_args()

    t0 = time.time()

    # Phase 1: Scan self_collected files
    self_collected_index, all_market_ids = scan_self_collected_files(args.self_collected_dir)
    if not self_collected_index:
        print("No usable self_collected files found.", file=sys.stderr)
        sys.exit(1)

    # Phase 1: Enrich from Gamma
    market_meta = await enrich_from_gamma(all_market_ids, parallel=args.parallel)
    if not market_meta:
        print("No markets enriched.", file=sys.stderr)
        sys.exit(1)

    # Group markets by game
    games = defaultdict(list)
    for mid, info in market_meta.items():
        if info["game_slug"]:  # skip markets with empty slug
            games[info["game_slug"]].append(mid)

    game_filter = set(args.games.split(",")) if args.games else None

    # Write games.csv (filtered if --games specified)
    if game_filter:
        filtered_meta = {mid: info for mid, info in market_meta.items()
                         if info["game_slug"] in game_filter}
    else:
        filtered_meta = {mid: info for mid, info in market_meta.items()
                         if info["game_slug"]}
    csv_rows = build_games_csv(filtered_meta, args.out_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped = 0
    empty = 0
    total_rows = 0

    game_list = sorted(games.items())
    for i, (game_slug, market_ids) in enumerate(game_list):
        if game_filter and game_slug not in game_filter:
            continue

        out_path = out_dir / f"{game_slug}.parquet"
        if not args.force and out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
            continue

        df = extract_game_events(game_slug, market_ids, self_collected_index, market_meta)

        if df is None or len(df) == 0:
            empty += 1
            continue

        df.to_parquet(out_path, index=False)
        total_rows += len(df)
        extracted += 1

        if extracted % 20 == 0 or extracted <= 3:
            elapsed = time.time() - t0
            print(f"[extract] {extracted} games extracted, {total_rows:,} rows, "
                  f"{elapsed:.0f}s elapsed", file=sys.stderr, flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s:", file=sys.stderr)
    print(f"  Games CSV: {args.out_csv} ({len(csv_rows)} rows)", file=sys.stderr)
    print(f"  Extracted: {extracted} games ({total_rows:,} total rows)", file=sys.stderr)
    print(f"  Skipped (exists): {skipped}", file=sys.stderr)
    print(f"  Empty (no events in window): {empty}", file=sys.stderr)
    print(f"  Output: {out_dir}", file=sys.stderr)

    if not args.no_filter:
        filter_self_collected(args.self_collected_dir, _DATA_DIR, Path(args.filter_out_dir))


if __name__ == "__main__":
    asyncio.run(main())
