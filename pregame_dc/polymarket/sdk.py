"""
Reusable Polymarket API infrastructure: WebSocket clients, order book tracking,
and data-loading helpers.

No dependency on py_clob_client — only stdlib + websockets.
"""

import asyncio
import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import websockets

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
WS_USER_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
WS_SPORTS_URL = "wss://sports-api.polymarket.com/ws"
CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137

# Log output paths — consumers assign these at startup (e.g.
# `polymarket.sdk.log_arb_path = "..."`). When None, log functions print only.
log_arb_path = None
order_log_path = None

PING_INTERVAL = 10
STATUS_INTERVAL = 60
RECONNECT_DELAY = 5

# --test mode config
MIN_ARB_SPREAD = 0.01      # 1 cent minimum spread to trigger test
MAX_GLOBAL_SPEND = 5.00    # $5 total budget
MIN_ORDER_SIZE = 5          # minimum order size in shares
FILL_TIMEOUT = 10.0        # seconds to wait for all legs to fill
DUMP_TIMEOUT = 10.0        # seconds to wait for dump sells
TEST_COOLDOWN = 30          # seconds between tests on same set
ARB_MIN_AGE = 10            # seconds an arb must persist before queuing test


def load_complete_sets(sport_names, sets_dir):
    """Load and merge complete sets from all enabled sports.

    Reads <sets_dir>/<sport>.json for each sport name.
    """
    sets_dir = Path(sets_dir)
    all_sets = []
    for name in sport_names:
        path = sets_dir / f"{name}.json"
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        with open(path) as f:
            all_sets.extend(json.load(f))
    return all_sets


def load_token_map(sport_names, markets_dir):
    """Load market_id -> (yes_token, no_token) from all enabled sports.

    Reads <markets_dir>/<sport>.csv for each sport name.
    """
    markets_dir = Path(markets_dir)
    tokens = {}
    for name in sport_names:
        path = markets_dir / f"{name}.csv"
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                mid = int(row["id"])
                tokens[mid] = (row["yes_token"], row["no_token"])
    return tokens


def build_lookups(complete_sets, token_map):
    """Build token_id -> (market_id, outcome) lookup and set structures."""
    # token_to_key: token_id_str -> (market_id, "YES"|"NO")
    token_to_key = {}
    for mid, (yes_tok, no_tok) in token_map.items():
        token_to_key[yes_tok] = (mid, "YES")
        token_to_key[no_tok] = (mid, "NO")

    # Resolve each complete set: list of (market_id, outcome) keys
    resolved_sets = []
    all_token_ids = set()
    for cs in complete_sets:
        keys = []
        for mid, outcome in cs["set"]:
            if mid not in token_map:
                break
            tok = token_map[mid][0] if outcome == "YES" else token_map[mid][1]
            keys.append((mid, outcome))
            all_token_ids.add(tok)
        else:
            resolved_sets.append({
                "game": cs["game"],
                "type": cs["type"],
                "keys": keys,
            })

    return token_to_key, resolved_sets, list(all_token_ids)


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def log_arb(msg, quiet=False):
    line = f"[{now_str()}] {msg}"
    if not quiet:
        print(line)
    if log_arb_path:
        with open(log_arb_path, "a") as f:
            f.write(line + "\n")


_order_log_file = None


def jlog(event, data=None, **fields):
    """Write a structured JSONL entry to the configured order_log_path."""
    entry = {"ts": datetime.now(timezone.utc).isoformat(), "event": event}
    if data is not None:
        entry["data"] = data
    entry.update(fields)
    global _order_log_file
    if _order_log_file is None:
        if not order_log_path:
            return
        _order_log_file = open(order_log_path, "a")
    _order_log_file.write(json.dumps(entry, default=str) + "\n")
    _order_log_file.flush()


class PolyWebSocket:
    """Base WS with app-level PING/PONG keepalive."""

    PING_MSG = "PING"

    def __init__(self):
        self.ws = None
        self._keepalive_task = None

    async def _keepalive(self):
        while True:
            await asyncio.sleep(PING_INTERVAL)
            try:
                await self.ws.send(self.PING_MSG)
            except websockets.ConnectionClosed:
                break

    def _start_keepalive(self):
        self._keepalive_task = asyncio.create_task(self._keepalive())

    def _cancel_keepalive(self):
        if self._keepalive_task:
            self._keepalive_task.cancel()


class UserWebSocket(PolyWebSocket):
    """Persistent WS connection for receiving trade/order confirmations.

    Supports two usage patterns:
    - connect() for short-lived usage (arb_monitor's burst pattern)
    - run() for long-lived usage with auto-reconnect (mover_leg)
    """

    def __init__(self, api_key, api_secret, api_passphrase,
                 reconnect_delay=RECONNECT_DELAY):
        super().__init__()
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.reconnect_delay = reconnect_delay
        self._reader_task = None
        self._queue = asyncio.Queue()
        self._stop_event = asyncio.Event()

    async def _connect_once(self):
        """Connect, authenticate, and start keepalive + reader."""
        self.ws = await websockets.connect(WS_USER_URL, ping_interval=None)
        auth_msg = {
            "auth": {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "passphrase": self.api_passphrase,
            },
            "markets": [],
            "assets_ids": [],
            "type": "user",
        }
        await self.ws.send(json.dumps(auth_msg))
        self._start_keepalive()
        print(f"[{now_str()}] User WS authenticated")

    async def connect(self):
        """Connect once (no auto-reconnect). For short-lived usage."""
        await self._connect_once()
        self._reader_task = asyncio.create_task(self._reader_loop())

    async def run(self):
        """Connect with auto-reconnect. Blocks until stop() is called."""
        while not self._stop_event.is_set():
            try:
                await self._connect_once()
                try:
                    async for raw in self.ws:
                        if self._stop_event.is_set():
                            return
                        if raw in ("PONG", "PING"):
                            continue
                        try:
                            jlog("ws_user", data=json.loads(raw))
                        except json.JSONDecodeError:
                            pass
                        await self._queue.put(raw)
                finally:
                    self._cancel_keepalive()

            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                if self._stop_event.is_set():
                    return
                print(f"[{now_str()}] User WS lost: {e}. "
                      f"Reconnecting in {self.reconnect_delay}s...")
                await asyncio.sleep(self.reconnect_delay)

    async def _reader_loop(self):
        """Read loop for connect() mode. Exits on disconnect."""
        try:
            async for raw in self.ws:
                if raw in ("PONG", "PING"):
                    continue
                try:
                    jlog("ws_user", data=json.loads(raw))
                except json.JSONDecodeError:
                    pass
                await self._queue.put(raw)
        except (websockets.ConnectionClosed, asyncio.CancelledError):
            pass
        await self._queue.put(None)

    def stop(self):
        """Signal the run loop to exit."""
        self._stop_event.set()

    async def wait_for_fills(self, order_ids, timeout):
        """Wait for MATCHED trade events for given order_ids.
        Returns dict of {order_id: filled_size} for orders that matched."""
        filled = {}  # order_id -> filled_size (float)
        remaining_ids = set(order_ids)
        deadline = time.monotonic() + timeout

        while remaining_ids:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                raw = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            if raw is None:
                break
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            items = data if isinstance(data, list) else [data]
            for item in items:
                status = (item.get("status") or item.get("orderStatus") or "").upper()
                if status != "MATCHED":
                    continue
                oid = item.get("taker_order_id") or item.get("orderID", "")
                if oid in remaining_ids:
                    size_str = item.get("size", "0")
                    filled[oid] = float(size_str) if size_str else 0.0
                    remaining_ids.discard(oid)

        return filled

    async def drain(self):
        """Drain buffered messages without blocking."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def close(self):
        self._stop_event.set()
        self._cancel_keepalive()
        if self._reader_task:
            self._reader_task.cancel()
        if self.ws:
            await self.ws.close()


class SportsWebSocket(PolyWebSocket):
    """Sports data WS with auto-reconnect. Lowercase ping/pong protocol."""

    PING_MSG = "ping"

    def __init__(self, game_ids=None, reconnect_delay=RECONNECT_DELAY):
        super().__init__()
        self.game_ids = set(game_ids) if game_ids else None
        self.reconnect_delay = reconnect_delay
        self.on_update = None
        self.on_connect = None
        self._periodics = []
        self._stop_event = asyncio.Event()

    def add_periodic(self, interval, callback):
        self._periodics.append((interval, callback))

    def stop(self):
        self._stop_event.set()

    async def run(self):
        """Connect and dispatch sports events. Reconnects on failure."""
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(WS_SPORTS_URL, ping_interval=None) as ws:
                    self.ws = ws

                    if self.on_connect:
                        await self.on_connect()

                    self._start_keepalive()
                    periodic_tasks = [
                        asyncio.create_task(self._run_periodic(interval, fn))
                        for interval, fn in self._periodics
                    ]

                    try:
                        async for raw in ws:
                            if self._stop_event.is_set():
                                return
                            if raw == "pong":
                                continue
                            if raw == "ping":
                                await ws.send("pong")
                                continue
                            try:
                                data = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                            if self.game_ids is not None and data.get("gameId") not in self.game_ids:
                                continue
                            if self.on_update:
                                self.on_update(data)
                    finally:
                        self._cancel_keepalive()
                        for t in periodic_tasks:
                            t.cancel()

            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                if self._stop_event.is_set():
                    return
                print(f"[{now_str()}] Sports WS lost: {e}. "
                      f"Reconnecting in {self.reconnect_delay}s...")
                await asyncio.sleep(self.reconnect_delay)

    @staticmethod
    async def _run_periodic(interval, fn):
        while True:
            await asyncio.sleep(interval)
            await fn()


class MarketWebSocket(PolyWebSocket):
    """Market data WS with auto-reconnect and periodic task management."""

    def __init__(self, token_ids, reconnect_delay=RECONNECT_DELAY):
        super().__init__()
        self.token_ids = token_ids
        self.reconnect_delay = reconnect_delay
        self.on_update = None       # callable(event: dict)
        self.on_connect = None      # async callable()
        self._periodics = []        # [(interval_sec, async_callable)]
        self._stop_event = asyncio.Event()

    def add_periodic(self, interval, callback):
        """Register an async callback to run every `interval` seconds."""
        self._periodics.append((interval, callback))

    def stop(self):
        """Signal the run loop to exit."""
        self._stop_event.set()

    @property
    def recv_queue_depth(self):
        try:
            return len(self.ws.recv_messages.frames)
        except AttributeError:
            return 0

    @property
    def recv_queue_paused(self):
        try:
            return self.ws.recv_messages.paused
        except AttributeError:
            return False

    @property
    def is_connected(self) -> bool:
        return self.ws is not None and self.ws.state == websockets.State.OPEN

    async def run(self):
        """Connect, subscribe, dispatch events. Reconnects on failure."""
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(WS_URL, ping_interval=None,
                                              max_size=10_000_000) as ws:
                    self.ws = ws

                    # Subscribe
                    await ws.send(json.dumps({
                        "assets_ids": self.token_ids,
                        "type": "market",
                        "custom_feature_enabled": True,
                    }))

                    # Notify caller (reset state, etc.)
                    if self.on_connect:
                        await self.on_connect()

                    # Start keepalive + periodic tasks
                    self._start_keepalive()
                    periodic_tasks = [
                        asyncio.create_task(self._run_periodic(interval, fn))
                        for interval, fn in self._periodics
                    ]

                    try:
                        async for raw in ws:
                            if self._stop_event.is_set():
                                return
                            if raw == "PONG":
                                continue
                            try:
                                msg = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                            events = msg if isinstance(msg, list) else [msg]
                            for evt in events:
                                self.on_update(evt)
                    finally:
                        self._cancel_keepalive()
                        for t in periodic_tasks:
                            t.cancel()

            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                if self._stop_event.is_set():
                    return
                print(f"[{now_str()}] Connection lost: {e}. "
                      f"Reconnecting in {self.reconnect_delay}s...")
                await asyncio.sleep(self.reconnect_delay)

    @staticmethod
    async def _run_periodic(interval, fn):
        while True:
            await asyncio.sleep(interval)
            await fn()


class BookState:
    """Order book state for a single (market_id, outcome) key.

    Each level stores {"price": float, "size": float, "ts": float} so that
    staleness is tracked per price level, not per key.
    """

    def __init__(self):
        self.asks = []   # sorted ascending by price
        self.bids = []   # sorted descending by price
        self.last_ts = 0.0  # max ts seen (for quick access)

    def per_game_data(self):
        """Return a deep copy."""
        s = BookState()
        s.asks = [dict(level) for level in self.asks]
        s.bids = [dict(level) for level in self.bids]
        s.last_ts = self.last_ts
        return s

    def apply_book(self, asks_raw, bids_raw, ts_sec):
        """Apply full book per_game_data. Each level applied independently by timestamp.
        Levels not in the per_game_data are tombstoned (size=0) rather than removed."""
        # Asks: start from per_game_data, keep existing levels that are newer
        new_asks = {}
        for a in asks_raw:
            price = float(a["price"])
            new_asks[price] = {"price": price, "size": float(a["size"]), "ts": ts_sec}

        for level in self.asks:
            if level["ts"] > ts_sec:
                new_asks[level["price"]] = level
            elif level["price"] not in new_asks:
                # Tombstone: per_game_data says this level doesn't exist at ts_sec
                new_asks[level["price"]] = {"price": level["price"], "size": 0.0, "ts": ts_sec}

        self.asks = sorted(new_asks.values(), key=lambda x: x["price"])

        # Bids: same logic
        new_bids = {}
        for b in bids_raw:
            price = float(b["price"])
            new_bids[price] = {"price": price, "size": float(b["size"]), "ts": ts_sec}

        for level in self.bids:
            if level["ts"] > ts_sec:
                new_bids[level["price"]] = level
            elif level["price"] not in new_bids:
                new_bids[level["price"]] = {"price": level["price"], "size": 0.0, "ts": ts_sec}

        self.bids = sorted(new_bids.values(), key=lambda x: x["price"], reverse=True)

        if ts_sec > self.last_ts:
            self.last_ts = ts_sec
        return True

    def apply_price_change(self, side, price, size, ts_sec):
        """Apply incremental price change. Only updates if newer than existing level.
        Size=0 tombstones the level rather than removing it."""
        price = float(price)
        size = float(size)

        if side == "SELL":
            for x in self.asks:
                if x["price"] == price:
                    if ts_sec < x["ts"]:
                        return False
                    x["size"] = size
                    x["ts"] = ts_sec
                    if ts_sec > self.last_ts:
                        self.last_ts = ts_sec
                    return True
            # New level — add it (even if size=0, as tombstone)
            self.asks.append({"price": price, "size": size, "ts": ts_sec})
            self.asks.sort(key=lambda x: x["price"])
            if ts_sec > self.last_ts:
                self.last_ts = ts_sec
            return True

        elif side == "BUY":
            for x in self.bids:
                if x["price"] == price:
                    if ts_sec < x["ts"]:
                        return False
                    x["size"] = size
                    x["ts"] = ts_sec
                    if ts_sec > self.last_ts:
                        self.last_ts = ts_sec
                    return True
            self.bids.append({"price": price, "size": size, "ts": ts_sec})
            self.bids.sort(key=lambda x: x["price"], reverse=True)
            if ts_sec > self.last_ts:
                self.last_ts = ts_sec
            return True

        return False


class MarketTracker:
    """Tracks order book state (bids and asks) for all subscribed tokens."""

    def __init__(self, token_to_key, all_token_ids):
        self.token_to_key = token_to_key
        self.all_token_ids = all_token_ids

        # Reverse lookup: (market_id, outcome) -> token_id_str
        self.key_to_token = {v: k for k, v in token_to_key.items()}

        # Per-key book state
        self.books = {}  # (market_id, outcome) -> BookState

        self.update_count = 0
        self.books_received = 0
        self.stale_rejected = 0  # level changes rejected due to older timestamp
        self.most_recent_ts = 0.0  # most recent event timestamp seen (global)
        self.out_of_order = 0  # events with timestamp older than most_recent_ts

        # Set of (market_id, outcome) keys that have changed since last consume
        self.dirty_keys = set()

    def reset(self):
        """Clear all book state. Call on WS reconnect to avoid stale data."""
        self.books.clear()
        self.dirty_keys.clear()

    def _get_or_create_book(self, key):
        book = self.books.get(key)
        if book is None:
            book = BookState()
            self.books[key] = book
        return book

    def get_best_ask(self, market_id, outcome):
        book = self.books.get((market_id, outcome))
        if not book:
            return 1.0
        for level in book.asks:
            if level["size"] > 0:
                return level["price"]
        return 1.0

    def get_best_ask_with_size(self, market_id, outcome):
        """Return (price, size) from best ask, or (1.0, 0.0) if book empty."""
        book = self.books.get((market_id, outcome))
        if not book:
            return (1.0, 0.0)
        for level in book.asks:
            if level["size"] > 0:
                return (level["price"], level["size"])
        return (1.0, 0.0)

    def get_best_bid(self, market_id, outcome):
        """Return best bid price, or 0.0 if bid book empty."""
        book = self.books.get((market_id, outcome))
        if not book:
            return 0.0
        for level in book.bids:
            if level["size"] > 0:
                return level["price"]
        return 0.0

    def get_best_bid_with_size_and_ts(self, market_id, outcome):
        """Return (price, size, ts) of the best bid level (highest price with
        size > 0), or None if the bid book is empty / all-tombstoned."""
        book = self.books.get((market_id, outcome))
        if not book:
            return None
        for level in book.bids:
            if level["size"] > 0:
                return (level["price"], level["size"], level["ts"])
        return None

    def get_best_ask_with_size_and_ts(self, market_id, outcome):
        """Return (price, size, ts) of the best ask level (lowest price with
        size > 0), or None if the ask book is empty / all-tombstoned."""
        book = self.books.get((market_id, outcome))
        if not book:
            return None
        for level in book.asks:
            if level["size"] > 0:
                return (level["price"], level["size"], level["ts"])
        return None

    def get_best_bid_with_size(self, market_id, outcome):
        """Return (price, size) from best bid, or (0.0, 0.0) if book empty."""
        book = self.books.get((market_id, outcome))
        if not book:
            return (0.0, 0.0)
        for level in book.bids:
            if level["size"] > 0:
                return (level["price"], level["size"])
        return (0.0, 0.0)

    def consume_dirty_keys(self):
        """Return and clear the set of changed (market_id, outcome) keys."""
        keys = self.dirty_keys
        self.dirty_keys = set()
        return keys

    def process_book(self, evt):
        """Handle full book per_game_data."""
        asset_id = evt.get("asset_id", "")
        key = self.token_to_key.get(asset_id)
        if not key:
            return

        ts = evt.get("timestamp")
        if ts is None:
            return
        ts_sec = int(ts) / 1000.0

        if ts_sec >= self.most_recent_ts:
            self.most_recent_ts = ts_sec
        else:
            self.out_of_order += 1

        book = self._get_or_create_book(key)
        if book.apply_book(evt.get("asks", []), evt.get("bids", []), ts_sec):
            self.books_received += 1
            self.dirty_keys.add(key)

    def process_price_change(self, evt):
        """Handle incremental price_change event."""
        ts = evt.get("timestamp")
        if ts is None:
            return
        ts_sec = int(ts) / 1000.0

        if ts_sec >= self.most_recent_ts:
            self.most_recent_ts = ts_sec
        else:
            self.out_of_order += 1
        changes = evt.get("price_changes", [])
        for pc in changes:
            asset_id = pc.get("asset_id", "")
            key = self.token_to_key.get(asset_id)
            if not key:
                continue

            book = self._get_or_create_book(key)
            side = pc.get("side", "")

            if book.apply_price_change(side, pc["price"], pc["size"], ts_sec):
                if side == "SELL":
                    self.dirty_keys.add(key)
            else:
                self.stale_rejected += 1

        self.update_count += 1
