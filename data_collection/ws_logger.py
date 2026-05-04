#!/usr/bin/env python3
"""Log raw Market and Sports WebSocket events to HDF5, print periodic summaries.

Output files are split into 24-hour UTC chunks: ws_log_<domain>_<YYYYMMDD>.h5
Restarting on the same UTC day appends to the existing file.

Encoding conventions
--------------------
- Timestamps: float64 seconds since Unix epoch (microsecond precision).
  - ``arrived_at``: wall-clock time the message was received, milliseconds
    since epoch (``time.time_ns() // 1_000_000``).
  - ``timestamp``: the event's own timestamp, milliseconds since epoch
    (passed through directly from the API string as float64).
- Token identity is encoded as two integers instead of raw strings:
  - ``market_idx`` (uint16): row index into ``/meta/markets`` (and the source
    CSV). Recoverable via ``markets[market_idx]``.
  - ``is_yes`` (uint8): 1 = yes token, 0 = no token.
- Side encoding (uint8):  BUY = 0, SELL = 1, BID = 0, ASK = 1.
  (BUY/BID share 0; SELL/ASK share 1.)
- Boolean fields (``live``, ``ended``) are stored as HDF5 booleans.
- Rare/complex events (``new_market``, ``market_resolved``) are stored as
  JSON blobs in a ``json_data`` vlen-string column.

HDF5 structure (column-per-dataset)
------------------------------------
Each leaf is a 1-D dataset. All resizable (``maxshape=(None,)``), gzip-4
compressed, chunked (20000 for price_change, 5000 for best_bid_ask/sports,
2560 for infrequent tables).

/meta/
    markets              [vlen str]  market IDs from CSV, ordered by index
    game_ids             [u8]        per-market game_id from CSV (0 if none);
                                     same length as ``markets``, indexed by market_idx
    connections/         (one row per WS connect/reconnect; gaps between
                          consecutive rows for the same source indicate disconnects)
        connected_at     [f8]        ms since epoch
        source           [vlen str]  "MARKET" or "SPORTS"

/market/price_change/    (~85 % of traffic, each WS event → 2 rows)
    arrived_at           [f8]
    timestamp            [f8]
    market_idx           [u2]
    is_yes               [u1]
    side                 [u1]        0 = BUY, 1 = SELL
    price                [f8]
    size                 [f8]
    best_bid             [f8]
    best_ask             [f8]

/market/best_bid_ask/    (~13 %)
    arrived_at           [f8]
    timestamp            [f8]
    market_idx           [u2]
    is_yes               [u1]
    best_bid             [f8]
    best_ask             [f8]
    spread               [f8]

/market/last_trade_price/  (<1 %)
    arrived_at           [f8]
    timestamp            [f8]
    market_idx           [u2]
    is_yes               [u1]
    price                [f8]
    size                 [f8]
    side                 [u1]        0 = BUY, 1 = SELL
    fee_rate_bps         [u2]
    transaction_hash     [vlen str]

/market/book_meta/       (~1 %, one row per snapshot)
    arrived_at           [f8]
    timestamp            [f8]
    market_idx           [u2]
    is_yes               [u1]
    book_id              [u8]        auto-increment FK to book_levels
    n_bids               [u2]
    n_asks               [u2]

/market/book_levels/     (one row per bid/ask level, joined via book_id)
    book_id              [u8]
    side                 [u1]        0 = BID, 1 = ASK
    price                [f8]
    size                 [f8]

/market/new_market/      (<0.3 %, JSON blob)
    arrived_at           [f8]
    timestamp            [f8]
    json_data            [vlen str]

/market/market_resolved/ (rare, JSON blob)
    arrived_at           [f8]
    timestamp            [f8]
    json_data            [vlen str]

/market/tick_size_change/ (rare)
    arrived_at           [f8]
    timestamp            [f8]
    market_idx           [u2]
    is_yes               [u1]
    old_tick_size        [f8]
    new_tick_size        [f8]

/sports/update/
    arrived_at           [f8]
    game_id              [u8]        numeric gameId from the payload
    league               [vlen str]  e.g. "nba", "nhl", "mlb", "spl"
    sport_type           [vlen str]  eventState.type, e.g. "basketball", "soccer"
    home_team            [vlen str]
    away_team            [vlen str]
    status               [vlen str]  e.g. "InProgress", "Final"
    live                 [bool]
    ended                [bool]
    score                [vlen str]
    period               [vlen str]
    elapsed              [vlen str]
    updated_at           [vlen str]  ISO8601 string from top-level updatedAt

/unknown/                (catch-all for unrecognized events, JSON blob)
    arrived_at           [f8]
    timestamp            [f8]
    json_data            [vlen str]

Usage
-----
    python ws_logger.py --soccer
    python ws_logger.py --nba -o ./logs --interval 5
    python ws_logger.py --soccer --flush-rows 10000 --flush-interval 30

Discovery is run automatically at startup and re-run every --refresh-hours
(default 1h) so newly listed markets (e.g. those that only appear within 48h
of kickoff) get picked up without restarting the logger.
"""

import argparse
import asyncio
import csv
import json
import os
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, date

import h5py
import numpy as np

from pregame_pca.polymarket.sdk import MarketWebSocket, SportsWebSocket
from pregame_pca import paths
from data_collection.discover_markets import discover_soccer_markets

# Soccer is the only domain pregame_pca supports. The "SPORTS" WS channel
# carries gameId-keyed live updates that we only filter on for sports
# domains; this set keeps the logic explicit (formerly came from
# strategies/arb_monitor/domains/__init__.py:SPORTS).
SPORTS = {"soccer"}

SUMMARY_INTERVAL = 60  # seconds
DEFAULT_REFRESH_HOURS = 1.0
DEFAULT_HOURS_AHEAD = 48
SIDE_MAP = {"BUY": 0, "SELL": 1, "BID": 0, "ASK": 1}

# --- Schema registry ---
# Each key is an HDF5 group path. Value is list of (col_name, h5py_dtype, chunk_size).
# chunk_size is per-dataset (frequent tables get larger chunks).
_str = h5py.string_dtype()

SCHEMAS = {
    "market/price_change": [
        ("arrived_at", "f8"), ("timestamp", "f8"),
        ("market_idx", "u2"), ("is_yes", "u1"), ("side", "u1"),
        ("price", "f8"), ("size", "f8"), ("best_bid", "f8"), ("best_ask", "f8"),
    ],
    "market/best_bid_ask": [
        ("arrived_at", "f8"), ("timestamp", "f8"),
        ("market_idx", "u2"), ("is_yes", "u1"),
        ("best_bid", "f8"), ("best_ask", "f8"), ("spread", "f8"),
    ],
    "market/last_trade_price": [
        ("arrived_at", "f8"), ("timestamp", "f8"),
        ("market_idx", "u2"), ("is_yes", "u1"),
        ("price", "f8"), ("size", "f8"), ("side", "u1"),
        ("fee_rate_bps", "u2"), ("transaction_hash", _str),
    ],
    "market/book_meta": [
        ("arrived_at", "f8"), ("timestamp", "f8"),
        ("market_idx", "u2"), ("is_yes", "u1"),
        ("book_id", "u8"), ("n_bids", "u2"), ("n_asks", "u2"),
    ],
    "market/book_levels": [
        ("book_id", "u8"), ("side", "u1"), ("price", "f8"), ("size", "f8"),
    ],
    "market/new_market": [
        ("arrived_at", "f8"), ("timestamp", "f8"), ("json_data", _str),
    ],
    "market/market_resolved": [
        ("arrived_at", "f8"), ("timestamp", "f8"), ("json_data", _str),
    ],
    "market/tick_size_change": [
        ("arrived_at", "f8"), ("timestamp", "f8"),
        ("market_idx", "u2"), ("is_yes", "u1"),
        ("old_tick_size", "f8"), ("new_tick_size", "f8"),
    ],
    "sports/update": [
        ("arrived_at", "f8"), ("game_id", "u8"),
        ("league", _str), ("sport_type", _str),
        ("home_team", _str), ("away_team", _str), ("status", _str),
        ("live", "bool"), ("ended", "bool"),
        ("score", _str), ("period", _str), ("elapsed", _str), ("updated_at", _str),
    ],
    "unknown": [
        ("arrived_at", "f8"), ("timestamp", "f8"), ("json_data", _str),
    ],
    "meta/connections": [
        ("connected_at", "f8"), ("source", _str),
    ],
}

# Chunk sizes: high-volume tables get bigger chunks
CHUNK_SIZES = defaultdict(lambda: 2560, {
    "market/price_change": 20000,
    "market/best_bid_ask": 5000,
    "sports/update": 5000,
})


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def csv_path_for(domain):
    # Only soccer is supported. The arg is kept so the rest of ws_logger
    # (which threads `domain` through the WS plumbing) doesn't need touching.
    if domain != "soccer":
        raise ValueError(f"unsupported domain: {domain!r} (only 'soccer' is supported)")
    return str(paths.MARKETS_CSV)


class MarketRegistry:
    """Tracks the running set of markets and their HDF5 indices.

    Indices are assigned the first time a market_id is seen and never
    reassigned, so existing event rows in the HDF5 file remain valid across
    discovery refreshes. New markets are appended at the end of the
    `market_ids` / `market_game_ids` lists. Markets that disappear from the
    CSV are kept in the registry (so historical event rows can still be
    resolved), but their tokens are dropped from `token_ids` so the WS no
    longer subscribes to them.

    `token_to_idx` is mutated in place so any holder of a reference (e.g. the
    EventLogger) sees additions automatically.
    """

    def __init__(self):
        self.market_ids: list[str] = []
        self.market_game_ids: list[int] = []
        self.market_id_to_idx: dict[str, int] = {}
        self.token_to_idx: dict[str, tuple[int, int]] = {}
        self.token_ids: list[str] = []
        self.game_ids: set[int] = set()

    def load_from_csv(self, path):
        """Read the CSV and reconcile against the existing registry.

        Returns dict with keys:
            new_indices       list[int]   indices appended this load
            tokens_changed    bool        True iff token_ids set changed
            game_ids_changed  bool        True iff game_ids set changed
            n_markets         int         total markets in CSV
        """
        new_indices: list[int] = []
        new_token_ids: list[str] = []
        new_game_ids: set[int] = set()

        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid = row["id"]
                yes_tok = row["yes_token"]
                no_tok = row["no_token"]
                raw_gid = row.get("game_id", "")
                gid = 0
                if raw_gid:
                    try:
                        gid = int(raw_gid)
                        new_game_ids.add(gid)
                    except ValueError:
                        pass

                idx = self.market_id_to_idx.get(mid)
                if idx is None:
                    idx = len(self.market_ids)
                    self.market_ids.append(mid)
                    self.market_game_ids.append(gid)
                    self.market_id_to_idx[mid] = idx
                    new_indices.append(idx)
                else:
                    if self.market_game_ids[idx] != gid and gid:
                        self.market_game_ids[idx] = gid

                self.token_to_idx[yes_tok] = (idx, 1)
                self.token_to_idx[no_tok] = (idx, 0)
                new_token_ids.append(yes_tok)
                new_token_ids.append(no_tok)

        tokens_changed = set(new_token_ids) != set(self.token_ids)
        game_ids_changed = new_game_ids != self.game_ids
        self.token_ids = new_token_ids
        self.game_ids = new_game_ids
        return {
            "new_indices": new_indices,
            "tokens_changed": tokens_changed,
            "game_ids_changed": game_ids_changed,
            "n_markets": len(new_token_ids) // 2,
        }


class HDF5Writer:
    """Buffered, columnar HDF5 writer with daily file rotation."""

    def __init__(self, output_dir, domain, registry,
                 flush_rows=5000, flush_interval=10.0):
        self._output_dir = output_dir
        self._domain = domain
        self._registry = registry
        self._flush_rows = flush_rows
        self._flush_interval = flush_interval
        os.makedirs(output_dir, exist_ok=True)

        self._buffers = {}  # group_path -> {col_name: list}
        self._total_buffered = 0
        self._last_flush = time.monotonic()
        self._next_book_id = 0
        self._meta_len = 0  # rows already written to /meta/markets

        self._h5 = None
        self._current_date = None
        self._open_file()

    def _file_path(self, dt):
        name = f"ws_log_{self._domain}_{dt.strftime('%Y%m%dT%H%M%S')}.h5"
        return os.path.join(self._output_dir, name)

    def _open_file(self):
        now = datetime.now(timezone.utc)
        self._current_date = now.date()
        path = self._file_path(now)
        self._h5 = h5py.File(path, "w")
        self._h5.attrs["domain"] = self._domain
        meta = self._h5.require_group("meta")
        n = len(self._registry.market_ids)
        meta.create_dataset(
            "markets",
            data=np.array(self._registry.market_ids, dtype=object),
            dtype=_str,
            maxshape=(None,),
            chunks=(max(n, 1),),
        )
        meta.create_dataset(
            "game_ids",
            data=np.array(self._registry.market_game_ids, dtype="u8"),
            maxshape=(None,),
            chunks=(max(n, 1),),
        )
        self._meta_len = n
        self._next_book_id = 0

    def sync_meta(self):
        """Append any newly-registered markets to /meta/markets and
        /meta/game_ids. Called after a discovery refresh extends the registry.
        """
        if self._h5 is None:
            return
        n = len(self._registry.market_ids)
        if n <= self._meta_len:
            return
        meta = self._h5["meta"]
        new_ids = self._registry.market_ids[self._meta_len:n]
        new_gids = self._registry.market_game_ids[self._meta_len:n]

        ds_markets = meta["markets"]
        ds_markets.resize(n, axis=0)
        ds_markets[self._meta_len:n] = np.array(new_ids, dtype=object)

        ds_gids = meta["game_ids"]
        ds_gids.resize(n, axis=0)
        ds_gids[self._meta_len:n] = np.array(new_gids, dtype="u8")

        self._meta_len = n

    def _rotate(self):
        """Close current file and open a new one for the new UTC day."""
        self.flush()
        self._h5.close()
        self._open_file()
        print(f"[{ts()}] Rotated to {self.file_path}")

    @property
    def file_path(self):
        return self._h5.filename

    def next_book_id(self):
        bid = self._next_book_id
        self._next_book_id += 1
        return bid

    def append(self, group_path, row_dict):
        """Buffer a single row. row_dict maps column names to scalar values."""
        today = datetime.now(timezone.utc).date()
        if today != self._current_date:
            self._rotate()

        if group_path not in self._buffers:
            schema = SCHEMAS[group_path]
            self._buffers[group_path] = {col: [] for col, _ in schema}

        buf = self._buffers[group_path]
        for col, val in row_dict.items():
            buf[col].append(val)
        self._total_buffered += 1
        self._maybe_flush()

    def _maybe_flush(self):
        if self._total_buffered >= self._flush_rows:
            self.flush()
        elif time.monotonic() - self._last_flush >= self._flush_interval:
            self.flush()

    def flush(self):
        """Write all buffered data to HDF5 datasets."""
        for group_path, columns in self._buffers.items():
            first_col = next(iter(columns.values()))
            n_new = len(first_col)
            if n_new == 0:
                continue

            schema = dict(SCHEMAS[group_path])
            chunk_size = CHUNK_SIZES[group_path]
            grp = self._h5.require_group(group_path)

            for col_name, values in columns.items():
                dtype = schema[col_name]
                if dtype is _str:
                    arr = np.array(values, dtype=object)
                elif dtype == "bool":
                    arr = np.array(values, dtype=bool)
                else:
                    arr = np.array(values, dtype=dtype)

                if col_name not in grp:
                    grp.create_dataset(
                        col_name, data=arr,
                        maxshape=(None,), chunks=(chunk_size,),
                        compression="gzip", compression_opts=4,
                    )
                else:
                    ds = grp[col_name]
                    old_len = ds.shape[0]
                    ds.resize(old_len + n_new, axis=0)
                    ds[old_len:] = arr

            # Clear buffers for this group
            for v in columns.values():
                v.clear()

        self._total_buffered = 0
        self._last_flush = time.monotonic()

    def close(self):
        if self._h5 is None:
            return
        self.flush()
        self._h5.close()
        self._h5 = None


class EventLogger:
    """Parses WS events into columnar rows and writes to HDF5 via HDF5Writer."""

    def __init__(self, writer, token_to_idx, summary_interval):
        self._writer = writer
        self._token_to_idx = token_to_idx
        self._interval = summary_interval
        self._total = 0
        self._window = 0
        self._reconnects = 0
        self._seen_sources = set()  # track initial connects vs reconnects
        self._errors = 0
        self._closed = False
        self._last_print = time.monotonic()

    def log(self, source, data):
        arrived = time.time_ns() // 1_000_000
        if source == "MARKET":
            event_type = data.get("event_type", "unknown")
            try:
                self._log_market(event_type, data, arrived)
            except Exception:
                self._errors += 1
        elif source == "SPORTS":
            event_type = "update"
            try:
                self._log_sports(data, arrived)
            except Exception:
                self._errors += 1

        self._total += 1
        self._window += 1
        self._maybe_print()

    def _resolve_token(self, asset_id):
        """Returns (market_idx, is_yes) or None if unknown."""
        return self._token_to_idx.get(asset_id)

    def _parse_ts(self, data):
        """Extract ms-epoch timestamp as float64."""
        raw = data.get("timestamp", "")
        return float(raw) if raw else 0.0

    def _log_market(self, event_type, data, arrived):
        if event_type == "price_change":
            self._log_price_change(data, arrived)
        elif event_type == "best_bid_ask":
            self._log_best_bid_ask(data, arrived)
        elif event_type == "last_trade_price":
            self._log_last_trade_price(data, arrived)
        elif event_type == "book":
            self._log_book(data, arrived)
        elif event_type == "new_market":
            self._log_json_blob("market/new_market", data, arrived)
        elif event_type == "market_resolved":
            self._log_json_blob("market/market_resolved", data, arrived)
        elif event_type == "tick_size_change":
            self._log_tick_size_change(data, arrived)
        else:
            self._log_json_blob("unknown", data, arrived)

    def _log_price_change(self, data, arrived):
        ts_sec = self._parse_ts(data)
        for pc in data.get("price_changes", []):
            tok = self._resolve_token(pc.get("asset_id", ""))
            if tok is None:
                continue
            market_idx, is_yes = tok
            self._writer.append("market/price_change", {
                "arrived_at": arrived,
                "timestamp": ts_sec,
                "market_idx": market_idx,
                "is_yes": is_yes,
                "side": SIDE_MAP.get(pc.get("side", ""), 0),
                "price": float(pc.get("price", 0)),
                "size": float(pc.get("size", 0)),
                "best_bid": float(pc.get("best_bid", 0)),
                "best_ask": float(pc.get("best_ask", 0)),
            })

    def _log_best_bid_ask(self, data, arrived):
        tok = self._resolve_token(data.get("asset_id", ""))
        if tok is None:
            return
        market_idx, is_yes = tok
        self._writer.append("market/best_bid_ask", {
            "arrived_at": arrived,
            "timestamp": self._parse_ts(data),
            "market_idx": market_idx,
            "is_yes": is_yes,
            "best_bid": float(data.get("best_bid", 0)),
            "best_ask": float(data.get("best_ask", 0)),
            "spread": float(data.get("spread", 0)),
        })

    def _log_last_trade_price(self, data, arrived):
        tok = self._resolve_token(data.get("asset_id", ""))
        if tok is None:
            return
        market_idx, is_yes = tok
        self._writer.append("market/last_trade_price", {
            "arrived_at": arrived,
            "timestamp": self._parse_ts(data),
            "market_idx": market_idx,
            "is_yes": is_yes,
            "price": float(data.get("price", 0)),
            "size": float(data.get("size", 0)),
            "side": SIDE_MAP.get(data.get("side", ""), 0),
            "fee_rate_bps": int(data.get("fee_rate_bps", 0)),
            "transaction_hash": data.get("transaction_hash", ""),
        })

    def _log_book(self, data, arrived):
        tok = self._resolve_token(data.get("asset_id", ""))
        if tok is None:
            return
        market_idx, is_yes = tok
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        book_id = self._writer.next_book_id()

        self._writer.append("market/book_meta", {
            "arrived_at": arrived,
            "timestamp": self._parse_ts(data),
            "market_idx": market_idx,
            "is_yes": is_yes,
            "book_id": book_id,
            "n_bids": len(bids),
            "n_asks": len(asks),
        })
        for level in bids:
            self._writer.append("market/book_levels", {
                "book_id": book_id,
                "side": 0,  # BID
                "price": float(level.get("price", 0)),
                "size": float(level.get("size", 0)),
            })
        for level in asks:
            self._writer.append("market/book_levels", {
                "book_id": book_id,
                "side": 1,  # ASK
                "price": float(level.get("price", 0)),
                "size": float(level.get("size", 0)),
            })

    def _log_tick_size_change(self, data, arrived):
        tok = self._resolve_token(data.get("asset_id", ""))
        if tok is None:
            return
        market_idx, is_yes = tok
        self._writer.append("market/tick_size_change", {
            "arrived_at": arrived,
            "timestamp": self._parse_ts(data),
            "market_idx": market_idx,
            "is_yes": is_yes,
            "old_tick_size": float(data.get("old_tick_size", 0)),
            "new_tick_size": float(data.get("new_tick_size", 0)),
        })

    def _log_json_blob(self, group_path, data, arrived):
        self._writer.append(group_path, {
            "arrived_at": arrived,
            "timestamp": self._parse_ts(data),
            "json_data": json.dumps(data),
        })

    def _log_sports(self, data, arrived):
        event_state = data.get("eventState") or {}
        self._writer.append("sports/update", {
            "arrived_at": arrived,
            "game_id": int(data.get("gameId", 0) or 0),
            "league": data.get("leagueAbbreviation", "") or "",
            "sport_type": event_state.get("type", "") or "",
            "home_team": data.get("homeTeam", "") or "",
            "away_team": data.get("awayTeam", "") or "",
            "status": data.get("status", "") or "",
            "live": bool(data.get("live", False)),
            "ended": bool(data.get("ended", False)),
            "score": data.get("score", "") or "",
            "period": data.get("period", "") or "",
            "elapsed": data.get("elapsed", "") or "",
            "updated_at": data.get("updatedAt", "") or "",
        })

    def log_connect(self, source):
        """Record a WS connection/reconnection event."""
        if source in self._seen_sources:
            self._reconnects += 1
        else:
            self._seen_sources.add(source)
        self._writer.append("meta/connections", {
            "connected_at": time.time_ns() // 1_000_000,
            "source": source,
        })

    def _maybe_print(self):
        now = time.monotonic()
        elapsed = now - self._last_print
        if elapsed < self._interval:
            return
        msg_per_sec = self._window / elapsed
        extra = []
        if self._reconnects:
            extra.append(f"reconnects: {self._reconnects}")
        if self._errors:
            extra.append(f"errors: {self._errors}")
        suffix = f" | {' | '.join(extra)}" if extra else ""
        print(f"[{ts()}] {msg_per_sec:.1f} msg/s ({self._total} total){suffix}")
        self._window = 0
        self._last_print = now

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._window > 0:
            self._last_print = 0
            self._maybe_print()
        self._writer.close()


def _run_discovery_sync(domain, hours_ahead):
    """Blocking discovery call — runs in a worker thread."""
    if domain != "soccer":
        raise ValueError(f"unsupported domain: {domain!r}")
    discover_soccer_markets(out_csv=paths.MARKETS_CSV, hours_ahead=hours_ahead)


async def _refresh_discovery(domain, hours_ahead, registry, writer):
    """Run discovery, reload the CSV into the registry, extend HDF5 meta.

    Returns the dict from `registry.load_from_csv` so the caller can decide
    whether to rebuild the WS connections.
    """
    print(f"[{ts()}] Running discovery (hours_ahead={hours_ahead})...")
    try:
        await asyncio.to_thread(_run_discovery_sync, domain, hours_ahead)
    except Exception as e:
        print(f"[{ts()}] Discovery failed: {e}")
        return None

    try:
        info = registry.load_from_csv(csv_path_for(domain))
    except Exception as e:
        print(f"[{ts()}] Reloading markets.csv failed: {e}")
        return None

    if info["new_indices"]:
        writer.sync_meta()

    print(f"[{ts()}] Discovery: {info['n_markets']} markets "
          f"(+{len(info['new_indices'])} new), "
          f"{len(registry.game_ids)} game_ids, "
          f"tokens_changed={info['tokens_changed']}, "
          f"games_changed={info['game_ids_changed']}")
    return info


async def run(domain, output_dir, interval, flush_rows, flush_interval,
              refresh_hours, hours_ahead, skip_initial_discovery):
    registry = MarketRegistry()

    csv_path = csv_path_for(domain)
    if not skip_initial_discovery:
        print(f"[{ts()}] Initial discovery for domain={domain} "
              f"(hours_ahead={hours_ahead})...")
        try:
            await asyncio.to_thread(_run_discovery_sync, domain, hours_ahead)
        except Exception as e:
            print(f"[{ts()}] Initial discovery failed: {e}")
            if not os.path.exists(csv_path):
                print(f"[{ts()}] No existing {csv_path}; exiting.")
                return

    if not os.path.exists(csv_path):
        print(f"[{ts()}] {csv_path} does not exist. "
              f"Run without --skip-initial-discovery, or run "
              f"`python -m data_collection.discover_markets` first.")
        return

    registry.load_from_csv(csv_path)

    writer = HDF5Writer(output_dir, domain, registry, flush_rows, flush_interval)
    logger = EventLogger(writer, registry.token_to_idx, interval)

    print(f"[{ts()}] Domain: {domain} | {len(registry.token_ids)} tokens "
          f"| {len(registry.game_ids)} games")
    print(f"[{ts()}] Logging to: {writer.file_path} | Summary every {interval}s "
          f"| Refresh every {refresh_hours:.2f}h")

    # Mutable handles so the refresh task can swap WS instances in/out.
    state = {
        "mws": None,
        "mws_task": None,
        "sws": None,
        "sws_task": None,
    }
    stop_event = asyncio.Event()
    refresh_lock = asyncio.Lock()

    async def stop_ws(ws, task):
        if ws is None:
            return
        ws.stop()
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                task.cancel()

    async def build_market_ws():
        await stop_ws(state["mws"], state["mws_task"])
        token_ids = list(registry.token_ids)
        if not token_ids:
            state["mws"] = None
            state["mws_task"] = None
            print(f"[{ts()}] MARKET WS not started (no tokens)")
            return
        mws = MarketWebSocket(token_ids)
        mws.on_update = lambda evt: logger.log("MARKET", evt)

        async def on_connect():
            logger.log_connect("MARKET")
            print(f"[{ts()}] MARKET WS connected, subscribed to {len(token_ids)} tokens")

        mws.on_connect = on_connect
        state["mws"] = mws
        state["mws_task"] = asyncio.create_task(mws.run())

    async def build_sports_ws():
        await stop_ws(state["sws"], state["sws_task"])
        if domain not in SPORTS or not registry.game_ids:
            state["sws"] = None
            state["sws_task"] = None
            reason = ("non-sport domain" if domain not in SPORTS
                      else "no game_ids in markets.csv")
            print(f"[{ts()}] Skipping Sports WS ({reason})")
            return
        game_ids = set(registry.game_ids)
        sws = SportsWebSocket(game_ids=game_ids)
        sws.on_update = lambda data: logger.log("SPORTS", data)

        async def on_connect():
            logger.log_connect("SPORTS")
            print(f"[{ts()}] SPORTS WS connected, filtering to {len(game_ids)} gameIds")

        sws.on_connect = on_connect
        state["sws"] = sws
        state["sws_task"] = asyncio.create_task(sws.run())

    await build_market_ws()
    await build_sports_ws()

    async def periodic_refresh():
        delay = max(60.0, refresh_hours * 3600.0)
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=delay)
                return
            except asyncio.TimeoutError:
                pass
            async with refresh_lock:
                info = await _refresh_discovery(domain, hours_ahead, registry, writer)
                if info is None:
                    continue
                if info["tokens_changed"]:
                    await build_market_ws()
                if info["game_ids_changed"]:
                    await build_sports_ws()

    refresh_task = asyncio.create_task(periodic_refresh())

    loop = asyncio.get_running_loop()

    def shutdown():
        print(f"\n[{ts()}] Shutting down...")
        stop_event.set()

    loop.add_signal_handler(signal.SIGINT, shutdown)

    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        refresh_task.cancel()
        await stop_ws(state["mws"], state["mws_task"])
        await stop_ws(state["sws"], state["sws_task"])
        logger.close()


def main():
    parser = argparse.ArgumentParser(description="WebSocket event logger (HDF5 + summary)")
    parser.add_argument("--output-dir", "-o", type=str, default=str(paths.RAW_WS_LOGS),
                        help=f"Directory for .h5 files (default: {paths.RAW_WS_LOGS})")
    parser.add_argument("--interval", "-i", type=float, default=SUMMARY_INTERVAL,
                        help=f"Summary print interval in seconds (default: {SUMMARY_INTERVAL})")
    parser.add_argument("--flush-rows", type=int, default=5000,
                        help="Flush buffer after this many rows (default: 5000)")
    parser.add_argument("--flush-interval", type=float, default=10.0,
                        help="Flush buffer after this many seconds (default: 10)")
    parser.add_argument("--refresh-hours", type=float, default=DEFAULT_REFRESH_HOURS,
                        help=f"Re-run discovery every N hours "
                             f"(default: {DEFAULT_REFRESH_HOURS})")
    parser.add_argument("--hours-ahead", type=int, default=DEFAULT_HOURS_AHEAD,
                        help=f"Discovery horizon in hours "
                             f"(default: {DEFAULT_HOURS_AHEAD})")
    parser.add_argument("--skip-initial-discovery", action="store_true",
                        help="Use the existing markets.csv instead of running "
                             "discovery at startup")
    args = parser.parse_args()

    # pregame_pca only supports soccer; the rest of ws_logger threads `domain`
    # through registry/HDF5 paths so we keep the variable but hardcode it.
    domain = "soccer"

    asyncio.run(run(
        domain, args.output_dir, args.interval, args.flush_rows,
        args.flush_interval, args.refresh_hours, args.hours_ahead,
        args.skip_initial_discovery,
    ))


if __name__ == "__main__":
    main()
