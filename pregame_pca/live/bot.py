"""Pregame PCA live trading bot.

At t = game_start - 25min, the bot:
  1. Reads top-of-book asks for the 24 canonical tokens from
     MarketTracker (an in-memory book reconstructed from the long-lived
     market WS subscription set up at discovery time; see fire_for_game).
  2. Computes rank-3 PCR predictions using saved (mu, sd_safe, beta_K).
  3. For each moneyline (slots 0/1/2 + NO sides) and totals (7/8/9/10 + NO)
     where 0.01 <= ask <= 0.99 AND pred - ask > 0.05, places a FOK buy of
     size = max(5, ceil(1.00 / price)), price = ceil(ask*100)/100.
  4. Captures every user-WS event for ~60s after placement to a per-order
     JSONL. Logs a one-line summary (with outcome=null) to orders_summary.jsonl.

Default mode: dry-run (computes everything but does not call post_order).
Pass --live to actually place orders. $20 USDC budget cap by default.

Outcomes are filled in later by `resolve_outcomes.py`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Force line-buffered stdout so heartbeat / per-fire status lines appear
# promptly when the bot is run with output piped (e.g. `bot.py 2>&1 | tee`).
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import requests
from dotenv import load_dotenv

# ---- Polymarket CLOB v2 monkey-patch ------------------------------------
# Around 2026-04-27, Polymarket migrated to CLOB v2, deploying new CTF
# Exchange contract addresses on Polygon mainnet. py_clob_client v0.34.6
# (latest as of 2026-05-03) still hard-codes the old addresses, so every
# `client.post_order(...)` returns
#     PolyApiException[status_code=400, error_message={'error': 'order_version_mismatch'}]
# (the EIP-712 domain's verifyingContract is signed against the old
# exchange but the server expects the new one).
#
# See: https://github.com/Polymarket/py-clob-client/issues/335 / 336 / 337
# New addresses from https://docs.polymarket.com/resources/contracts.md:
#   CTF Exchange:           0xE111180000d2663C0091e4f400237545B87B996B
#   Neg Risk CTF Exchange:  0xe2222d279d744050d28e00520010520000310F59
# Once py-clob-client publishes a v2-aware release, this patch can be removed.
import py_clob_client.config as _pcc_config
from py_clob_client.clob_types import ContractConfig as _ContractConfig

_CLOB_V2_EXCHANGE = "0xE111180000d2663C0091e4f400237545B87B996B"
_CLOB_V2_NEG_RISK_EXCHANGE = "0xe2222d279d744050d28e00520010520000310F59"


def _patched_get_contract_config(chainID: int, neg_risk: bool = False):
    if chainID == 137:
        if neg_risk:
            return _ContractConfig(
                exchange=_CLOB_V2_NEG_RISK_EXCHANGE,
                collateral="0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
                conditional_tokens="0x4D97DCd97eC945f40cF65F87097ACe5EA0476045",
            )
        return _ContractConfig(
            exchange=_CLOB_V2_EXCHANGE,
            collateral="0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
            conditional_tokens="0x4D97DCd97eC945f40cF65F87097ACe5EA0476045",
        )
    # Fall through to the upstream default for non-mainnet chains (e.g. Amoy 80002)
    return _pcc_config.get_contract_config.__wrapped__(chainID, neg_risk) if hasattr(
        _pcc_config.get_contract_config, "__wrapped__"
    ) else _pcc_config._original_get_contract_config(chainID, neg_risk)


_pcc_config._original_get_contract_config = _pcc_config.get_contract_config
_pcc_config.get_contract_config = _patched_get_contract_config
# py_clob_client.client and py_clob_client.order_builder.builder both already
# imported get_contract_config by name at module-load time; patch those refs too.
import py_clob_client.client as _pcc_client_mod
import py_clob_client.order_builder.builder as _pcc_builder_mod
_pcc_client_mod.get_contract_config = _patched_get_contract_config
_pcc_builder_mod.get_contract_config = _patched_get_contract_config

# Second part of the v2 patch: bump the EIP-712 domain `version` field from
# "1" to "2". The new exchange contracts use version "2" in their EIP712Domain.
# This lives inside py_order_utils.builders.base_builder._get_domain_separator,
# which is called from BaseBuilder.__init__ each time a new OrderBuilder is
# constructed (and OrderBuilder is built fresh on every create_order call).
import py_order_utils.builders.base_builder as _pou_base
from poly_eip712_structs import make_domain as _make_domain


def _patched_get_domain_separator(self, chain_id, verifying_contract):
    return _make_domain(
        name="Polymarket CTF Exchange",
        version="2",
        chainId=str(chain_id),
        verifyingContract=verifying_contract,
    )


_pou_base.BaseBuilder._original_get_domain_separator = _pou_base.BaseBuilder._get_domain_separator
_pou_base.BaseBuilder._get_domain_separator = _patched_get_domain_separator
# ---- end CLOB v2 monkey-patch -------------------------------------------

import pregame_pca.polymarket.sdk as sdk
from pregame_pca.polymarket.sdk import UserWebSocket, jlog, now_str
from pregame_pca.polymarket.redundant_tracker import RedundantMarketTracker
from pregame_pca.discovery.soccer import build_constraints, fetch_markets
from pregame_pca.constants import MARKET_LABELS
from pregame_pca.discovery.slots import standard_slot_map
from pregame_pca.live.clob_v2 import build_signed_order_v2
from pregame_pca import paths

# ---- constants ----------------------------------------------------------

LOGS_DIR = paths.LOGS_DIR
WS_EVENTS_DIR = paths.WS_EVENTS_DIR
ORDERS_LOG = paths.ORDERS_SUMMARY
KICKOFFS_LOG = paths.KICKOFFS_LOG
STATE_FILE = paths.STATE_FILE
WS_MASTER_LOG = paths.WS_MASTER_LOG
DEFAULT_MODEL = paths.MODEL_T_25MIN

ASK_LO, ASK_HI = 0.01, 0.99
SPREAD_FLOOR = 0.005          # floor for book_spread when computing ratio
DEFAULT_THRESHOLD = 0.05
DEFAULT_RULE = "edge"         # "edge" → fire when edge > thr; "ratio" → fire when edge/book_spread > thr
DEFAULT_T_OFFSET_S = -1500          # -25 min
DEFAULT_BUDGET = 20.0
DEFAULT_HOURS_AHEAD = 48

# Per-fire post-placement window for capturing user-WS events to per-order log.
WS_RECORD_DURATION_S = 60.0
# Re-discover upcoming games every N seconds (1 hour).
DISCOVERY_REFRESH_S = 3600
# Spacing between sequential fires within a single game (avoid REST hammer).
INTRA_GAME_FIRE_DELAY_S = 1.0
# Heartbeat status line cadence.
HEARTBEAT_INTERVAL_S = 60

MONEYLINE_SLOTS_Y = [0, 1, 2]
TOTALS_SLOTS_Y = [7, 8, 9, 10]
# Trading both YES and NO sides; NO side has slot index +12.
CANDIDATE_SLOTS = (
    MONEYLINE_SLOTS_Y + TOTALS_SLOTS_Y +
    [s + 12 for s in MONEYLINE_SLOTS_Y + TOTALS_SLOTS_Y]
)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slot_label(slot: int) -> tuple[str, str, str]:
    """Return (market_type, market_label, side)."""
    base = slot % 12
    side = "Y" if slot < 12 else "N"
    if base in (0, 1, 2):
        mt = "moneyline"
    elif base in (3, 4, 5, 6):
        mt = "spread"
    elif base in (7, 8, 9, 10):
        mt = "totals"
    else:
        mt = "btts"
    return mt, MARKET_LABELS[base], side


# ---- per-order WS event recorder ---------------------------------------

class WsRecorder:
    """Captures user-WS events for a fixed duration, writing each as a JSON
    line to its own file. The dispatcher calls `push()` for each event the
    user-WS receives during the recorder's lifetime."""

    def __init__(self, path: Path, label: str, duration_s: float):
        self.path = path
        self.label = label
        self.deadline = time.monotonic() + duration_s
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a")
        self._open = True

    def is_open(self) -> bool:
        return self._open and time.monotonic() < self.deadline

    def push(self, event):
        if not self._open:
            return
        rec = {"ts": _now_utc_iso(), "label": self.label, "event": event}
        self._file.write(json.dumps(rec, default=str) + "\n")
        self._file.flush()

    def close(self):
        if self._open:
            self._file.close()
            self._open = False


# ---- model loader ------------------------------------------------------

class Model:
    def __init__(self, npz_path: Path):
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {npz_path}. "
                f"Run `python -m strategies.pregame_pca.live.fit_save_model` first."
            )
        z = np.load(npz_path)
        self.mu = z["mu"]                 # (24,)
        self.sd_safe = z["sd_safe"]       # (24,)
        self.beta_K = z["beta_K"]         # (24, 25)
        self.t_target = float(z["T_TARGET"])
        self.k = int(z["K"])
        self.train_n = int(z["train_n"])
        # Optional: impute-only variant ships impute_values (24,). When
        # present, ask cells equal to 1.0 (MarketTracker's "no book" sentinel)
        # are replaced with the corresponding training-set column mean before
        # standardising. Older baseline npz files have no such key and fall
        # back to feeding 1.0 sentinels through unchanged.
        self.impute_values = (
            z["impute_values"] if "impute_values" in z.files else None
        )

    def predict(self, asks_24: np.ndarray) -> np.ndarray:
        """asks_24: (24,) raw ask vector in canonical order. Returns (24,) preds."""
        x = asks_24
        if self.impute_values is not None:
            x = np.where(x == 1.0, self.impute_values, x)
        z = (x - self.mu) / self.sd_safe
        feats = np.concatenate([z, [1.0]])
        return self.beta_K @ feats


# ---- canonical-slot helpers --------------------------------------------

def _canonical_token_ids(game_markets) -> list[str] | None:
    """Return [yes_t_0, yes_t_1, ..., yes_t_11, no_t_0, ..., no_t_11] or None.

    Returns None if the game doesn't have all 12 canonical slots.
    """
    type_order = {"moneyline": 0, "spread": 1, "totals": 2, "btts": 3}
    sorted_markets = sorted(
        game_markets, key=lambda m: (type_order.get(m["type"], 99), m["id"])
    )
    result = build_constraints(sorted_markets)
    if result is None:
        return None
    n, _, _, slot_info = result
    slot_map = standard_slot_map(slot_info)
    if len(slot_map) != 12:
        return None

    yes_tokens = [sorted_markets[slot_map[s]]["yes_token"] for s in range(12)]
    no_tokens = [sorted_markets[slot_map[s]]["no_token"] for s in range(12)]
    return yes_tokens + no_tokens


def _canonical_market_ids(game_markets) -> list[int] | None:
    """Return market_id per canonical slot 0..11 (single per slot, regardless of side)."""
    type_order = {"moneyline": 0, "spread": 1, "totals": 2, "btts": 3}
    sorted_markets = sorted(
        game_markets, key=lambda m: (type_order.get(m["type"], 99), m["id"])
    )
    result = build_constraints(sorted_markets)
    if result is None:
        return None
    n, _, _, slot_info = result
    slot_map = standard_slot_map(slot_info)
    if len(slot_map) != 12:
        return None
    return [int(sorted_markets[slot_map[s]]["id"]) for s in range(12)]


# ---- bot ---------------------------------------------------------------

class PregamePCABot:
    def __init__(self, args):
        self.args = args
        self.live = bool(args.live)
        self.budget = float(args.budget)
        self.threshold = float(args.threshold)
        self.rule = str(args.rule)
        if self.rule not in ("edge", "ratio"):
            raise SystemExit(f"--rule: must be 'edge' or 'ratio', got {self.rule!r}")
        self.t_offset = int(args.time_seconds)

        # Market-type filter (comma-separated, e.g. "moneyline" or "moneyline,totals")
        wanted = [m.strip() for m in args.markets.split(",") if m.strip()]
        slot_groups = {
            "moneyline": MONEYLINE_SLOTS_Y,
            "totals":    TOTALS_SLOTS_Y,
        }
        for m in wanted:
            if m not in slot_groups:
                raise SystemExit(f"--markets: unknown type {m!r} (valid: {list(slot_groups)})")
        yes_slots = sum((slot_groups[m] for m in wanted), [])
        self.candidate_slots = yes_slots + [s + 12 for s in yes_slots]
        print(f"[init] market filter: {wanted}  candidate_slots={self.candidate_slots}")

        self.model = Model(Path(args.model))
        if abs(self.model.t_target - self.t_offset) > 1e-6:
            print(f"[warn] model.T_TARGET={self.model.t_target} but bot --time-seconds={self.t_offset}; "
                  f"using bot value but predictions may be miscalibrated")

        # Logging & state
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        WS_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        sdk.order_log_path = str(WS_MASTER_LOG)
        # Set of game slugs whose kickoff window we've already processed
        # (one kickoff -> N>=0 trades). Persisted across sessions.
        self._kickoff_slugs: set[str] = self._load_state()

        # Polymarket client (constructed lazily so dry-run can skip auth)
        self.client = None
        self.user_ws: UserWebSocket | None = None

        # WS event broadcasting
        self._recorders: list[WsRecorder] = []
        self._dispatcher_task: asyncio.Task | None = None

        # Discovery state
        self._scheduled_keys: set[str] = set()    # game_slug to avoid double-scheduling
        self._fire_tasks: list[asyncio.Task] = []
        self._fire_at: dict[str, float] = {}      # slug -> unix fire_at (for heartbeat)
        # game_slug -> game_info dict; maintained for the lifetime of the
        # game's pre-fire window so we can rebuild the market-WS subscription
        # set on every discovery refresh.
        self._upcoming_games: dict[str, dict] = {}

        # Long-lived market WebSocket: subscribed to every upcoming game's
        # 24 tokens at all times so book per_game_data are pre-warmed by the
        # time any kickoff fires. Replaced (disconnect+reconnect with new
        # token set) whenever discovery picks up newly listed games.
        # Wrapped in RedundantMarketTracker so a peer connection seeds the
        # book on each reconnect — closes the partial-book gap that arises
        # when Polymarket fails to re-send book snapshots after a drop.
        self._market_tracker: RedundantMarketTracker | None = None
        self._market_token_set: set[str] = set()

        # Spend & trade-count tracking (this session only)
        self.total_spent = 0.0
        self.n_trades_attempted = 0   # rows written to orders_summary.jsonl

        # Concurrency guards
        # Cap simultaneous kickoff processing to avoid REST hammer when many
        # games kick off at the same minute (e.g. EPL slate at 13:30).
        self._game_sem = asyncio.Semaphore(3)
        # Serialize budget check + reservation so concurrent fires can't both
        # pass the cap and overspend.
        self._budget_lock = asyncio.Lock()
        # py_clob_client caches per-token contract config (neg-risk, exchange
        # address) lazily on first create_order call. Concurrent create_orders
        # across different tokens can race on this cache and trigger
        # `order_version_mismatch` 400s. Serialize create+post per order.
        # If `order_version_mismatch` errors recur even with this lock in
        # place, the next thing to try is pre-warming the cache at fire time
        # by calling `client.get_neg_risk(token_id)` for each of the 24
        # canonical tokens (in serial) before any create_order. That populates
        # the SDK cache via single-threaded REST calls and eliminates the race
        # entirely.
        self._clob_lock = asyncio.Lock()

    # ---- state file ----

    def _load_state(self) -> set[str]:
        if STATE_FILE.exists():
            try:
                d = json.loads(STATE_FILE.read_text())
                # Backward-compat: prior versions used "fired_keys"
                return set(d.get("kickoff_slugs", d.get("fired_keys", [])))
            except json.JSONDecodeError:
                return set()
        return set()

    def _save_state(self):
        STATE_FILE.write_text(
            json.dumps({"kickoff_slugs": sorted(self._kickoff_slugs)})
        )

    # ---- credentials & client ----

    def _ensure_client(self):
        if self.client is not None:
            return
        load_dotenv()
        pk = os.getenv("POLYMARKET_PRIVATE_KEY")
        funder = os.getenv("POLYMARKET_FUNDER_ADDRESS")
        if not pk or not funder:
            raise RuntimeError("POLYMARKET_PRIVATE_KEY and POLYMARKET_FUNDER_ADDRESS must be set in .env")
        from py_clob_client.client import ClobClient
        self.client = ClobClient(
            sdk.CLOB_HOST, key=pk, chain_id=sdk.CHAIN_ID,
            signature_type=2, funder=funder,
        )
        creds = self.client.create_or_derive_api_creds()
        self.client.set_api_creds(creds)
        self._creds = creds
        print(f"[{now_str()}] CLOB client ready (funder={funder[:10]}...)")

    async def _ensure_user_ws(self):
        if self.user_ws is not None:
            return
        self._ensure_client()
        self.user_ws = UserWebSocket(
            self._creds.api_key, self._creds.api_secret, self._creds.api_passphrase,
        )
        # Run forever in the background; jlog() automatically tees every event
        # to WS_MASTER_LOG.
        asyncio.create_task(self.user_ws.run())
        # Dispatcher pulls from user_ws._queue and broadcasts to recorders.
        self._dispatcher_task = asyncio.create_task(self._dispatch_user_ws())

    async def _dispatch_user_ws(self):
        """Pull each event from user_ws._queue and push to all open recorders."""
        assert self.user_ws is not None
        while True:
            raw = await self.user_ws._queue.get()
            if raw is None:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            # Drop closed recorders
            self._recorders = [r for r in self._recorders if r.is_open()]
            for rec in self._recorders:
                try:
                    rec.push(event)
                except Exception as e:
                    print(f"[warn] recorder {rec.label} push failed: {e}")

    # ---- discovery + scheduling ----

    async def discover_and_schedule(self):
        """Fetch upcoming games, build canonical token lists, schedule firings."""
        try:
            markets = await asyncio.to_thread(
                fetch_markets, hours_ahead=self.args.hours_ahead, hours_behind=0,
            )
        except Exception as e:
            print(f"[{now_str()}] discovery failed: {e}")
            return

        # Group by game
        games: dict[str, list[dict]] = {}
        for m in markets:
            games.setdefault(m["game"], []).append(m)

        new_count = 0
        skip_old = 0
        skip_no_slots = 0
        skip_no_start = 0
        skip_already_done = 0
        for slug, game_markets in games.items():
            if slug in self._scheduled_keys:
                # Already scheduled this session — quietly skip (refresh case)
                continue
            if slug in self._kickoff_slugs:
                # Already fired this or a previous session
                skip_already_done += 1
                continue

            tokens = _canonical_token_ids(game_markets)
            if tokens is None:
                skip_no_slots += 1
                continue
            mids = _canonical_market_ids(game_markets)

            start_ts = None
            for m in game_markets:
                if m.get("start_time_ts"):
                    start_ts = int(m["start_time_ts"])
                    break
            if start_ts is None:
                skip_no_start += 1
                continue

            fire_at = start_ts + self.t_offset
            if fire_at < time.time() - 60:
                # game's fire time already passed
                skip_old += 1
                continue

            self._scheduled_keys.add(slug)
            self._fire_at[slug] = float(fire_at)
            game_id = next((m.get("game_id") for m in game_markets if m.get("game_id")), None)
            game_info = {
                "slug": slug,
                "game_id": game_id,
                "start_ts": start_ts,
                "fire_at": fire_at,
                "tokens_24": tokens,
                "market_ids_12": mids,
            }
            self._upcoming_games[slug] = game_info
            t = asyncio.create_task(self._sleep_then_fire(game_info))
            self._fire_tasks.append(t)
            new_count += 1

        print(f"[{now_str()}] discovery: {len(games)} games, "
              f"{new_count} newly scheduled, "
              f"skipped {skip_no_slots} (incomplete slot set), "
              f"{skip_old} (past fire time), "
              f"{skip_no_start} (no start_time), "
              f"{skip_already_done} (kickoff already processed in earlier session)")

        # Refresh the long-lived market WS subscription with the union of all
        # tokens across every upcoming-but-unfired game.
        await self._refresh_market_ws()

    async def _sleep_then_fire(self, game_info):
        delay = max(0.0, game_info["fire_at"] - time.time())
        slug = game_info["slug"]
        if delay > 0:
            print(f"[{now_str()}] scheduled {slug}: fire in {delay:.0f}s "
                  f"(start_ts={game_info['start_ts']})")
            await asyncio.sleep(delay)
        if slug in self._kickoff_slugs:
            return
        try:
            await self.fire_for_game(game_info)
        except Exception as e:
            print(f"[{now_str()}] fire_for_game({slug}) raised: {e}")
            jlog("fire_error", game_slug=slug, error=str(e))
        finally:
            self._kickoff_slugs.add(slug)
            self._upcoming_games.pop(slug, None)
            self._save_state()

    # ---- long-lived market WS ----

    async def _refresh_market_ws(self):
        """Refresh the long-lived market-WS subscription so it covers every
        upcoming-but-unfired game's (YES + NO) tokens.

        Polymarket's market WS has no incremental subscribe, so adding tokens
        requires disconnect+reconnect. **Critically**, every reconnect triggers
        a `tracker.reset()` (to avoid stale book state per CLAUDE.md), which
        drops all in-memory books and forces them to re-per_game_data from the
        server — taking 10s+ for 1700+ tokens. So:

          - If the new token set is a strict subset of what's already
            subscribed (i.e. some games fired out, no new games to add),
            DO NOT rebuild. Keep the existing connection alive — the extra
            subscription to already-closed markets is free. This avoids
            wiping our pre-warmed books right before the next kickoff cluster
            (the cause of the 13:35 / 14:35 / 15:35 abstain bursts).
          - Only rebuild when `new_set` contains tokens we're not yet
            subscribed to.
          - Update token_to_key on the existing tracker either way so newly-
            discovered games' (market_id, outcome) lookups work even without
            a rebuild.
        """
        # Build the new token set + token_to_key from upcoming games.
        new_tokens: list[str] = []
        new_token_to_key: dict[str, tuple[int, bool]] = {}
        for slug, gi in self._upcoming_games.items():
            for s in range(12):
                t_yes = gi["tokens_24"][s]
                t_no = gi["tokens_24"][s + 12]
                new_token_to_key[t_yes] = (int(gi["market_ids_12"][s]), True)
                new_token_to_key[t_no] = (int(gi["market_ids_12"][s]), False)
                new_tokens.append(t_yes)
                new_tokens.append(t_no)

        new_set = set(new_tokens)

        # Subset case: every new token is already subscribed → no rebuild.
        # Just keep the tracker's token_to_key fresh in case anything has
        # been added since last refresh (safe no-op when nothing new).
        if (
            self._market_tracker is not None
            and new_set.issubset(self._market_token_set)
        ):
            await self._market_tracker.update_token_to_key(new_token_to_key)
            return

        # Tear down existing tracker if any
        if self._market_tracker is not None:
            await self._market_tracker.stop()
            self._market_tracker = None

        if not new_tokens:
            self._market_token_set = set()
            return

        # Build new redundant tracker (manages K WSes + K MarketTrackers).
        self._market_tracker = RedundantMarketTracker(
            token_to_key=new_token_to_key,
            all_token_ids=new_tokens,
            k=2,
        )
        self._market_token_set = new_set
        await self._market_tracker.start()
        print(f"[{now_str()}] MARKET WS subscribed to {len(new_tokens)} tokens "
              f"({len(self._upcoming_games)} games)")

    async def _wait_for_books(self, slug, market_ids_12, timeout_s=10.0) -> bool:
        """At fire time, wait until both (YES) and (NO) books for all 12 slots
        of this game have arrived in the tracker. Returns True if all 24 are
        present before `timeout_s`, else False (caller MUST skip the kickoff).
        """
        if self._market_tracker is None:
            return False

        def all_present() -> bool:
            for s in range(12):
                mid = int(market_ids_12[s])
                if (mid, True) not in self._market_tracker.books:
                    return False
                if (mid, False) not in self._market_tracker.books:
                    return False
            return True

        if all_present():
            return True
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            await asyncio.sleep(0.2)
            if all_present():
                return True
        return False

    # ---- per-game firing ----

    async def fire_for_game(self, game_info):
        slug = game_info["slug"]
        # Cap concurrent kickoff processing.
        async with self._game_sem:
            tokens = game_info["tokens_24"]
            print(f"[{now_str()}] firing for game {slug} (start_ts={game_info['start_ts']})")

            self._ensure_client()
            if self.live:
                await self._ensure_user_ws()

            # Books are maintained continuously by the long-lived market WS
            # (set up at discovery time). At fire time, just wait briefly for
            # all 24 to be present (they almost always are), then read.
            # Fail-loud if any book is missing — silently defaulting to ask=1.0
            # would brick the model input (see 2026-05-03 sea-juv-ver incident).
            market_ids_12 = game_info["market_ids_12"]
            ready = await self._wait_for_books(slug, market_ids_12, timeout_s=10.0)
            if not ready or self._market_tracker is None:
                jlog("fire_skipped", game_slug=slug, reason="ws_books_incomplete")
                print(f"[{now_str()}]   {slug} skipped: not all 24 books in tracker")
                return

            asks_clean = np.empty(24, dtype=np.float64)
            sizes_clean = np.empty(24, dtype=np.float64)
            for s in range(12):
                yp, ys = self._market_tracker.get_best_ask_with_size(int(market_ids_12[s]), True)
                np_p, np_s = self._market_tracker.get_best_ask_with_size(int(market_ids_12[s]), False)
                asks_clean[s]      = yp;   sizes_clean[s]      = ys
                asks_clean[s + 12] = np_p; sizes_clean[s + 12] = np_s
            pred = self.model.predict(asks_clean)

            # Per-slot book spread: ask_yes + ask_no - 1, symmetric across the
            # YES/NO pair so we emit one value per market.
            book_spread_12 = [
                float(asks_clean[s] + asks_clean[s + 12] - 1.0) for s in range(12)
            ]

            # Persist the full kickoff snapshot — one line per game, whether
            # or not it produces any fires.
            kickoff_record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "game_slug": slug,
                "start_ts": int(game_info["start_ts"]),
                "t_offset_s": self.t_offset,
                "rule": self.rule,
                "rule_threshold": self.threshold,
                "asks": asks_clean.tolist(),
                "ask_sizes": sizes_clean.tolist(),
                "pred": pred[:12].tolist(),
                "book_spread": book_spread_12,
            }
            with open(KICKOFFS_LOG, "a") as f:
                f.write(json.dumps(kickoff_record) + "\n")

            # Snapshot of the inputs / model output for this kickoff.
            # Header line: short market labels, aligned 5-char wide to match
            # the 5-char "0.000" float format with 1-space separators.
            short_labels = [
                "A_win", "B_win", "Draw ",
                "A-1.5", "B-1.5", "A-2.5", "B-2.5",
                "O 1.5", "O 2.5", "O 3.5", "O 4.5",
                "BTTS ",
            ]
            header = " ".join(short_labels)
            asks_yes_str = " ".join(f"{a:.3f}" for a in asks_clean[:12])
            asks_no_str = " ".join(f"{a:.3f}" for a in asks_clean[12:])
            preds_str = " ".join(f"{p:.3f}" for p in pred[:12])
            print(f"[{now_str()}]   {slug} markets:    {header}")
            print(f"[{now_str()}]   {slug}    ask(YES): {asks_yes_str}")
            print(f"[{now_str()}]   {slug}    ask(NO):  {asks_no_str}")
            print(f"[{now_str()}]   {slug}    pred:     {preds_str}")

            # Find candidate fires. Score depends on --rule:
            #   "edge"  → score = edge,                                fire when score > threshold
            #   "ratio" → score = edge / max(book_spread, SPREAD_FLOOR), fire when score > threshold
            # book_spread is computed per-slot from ask_yes + ask_no - 1.
            candidates = []
            for slot in self.candidate_slots:
                ask = asks_clean[slot]
                if not (ASK_LO <= ask <= ASK_HI):
                    continue
                edge = float(pred[slot] - ask)
                ask_complement = float(asks_clean[(slot + 12) % 24])
                book_spread = ask + ask_complement - 1.0
                if self.rule == "ratio":
                    score = edge / max(book_spread, SPREAD_FLOOR)
                else:
                    score = edge
                if score <= self.threshold:
                    continue
                candidates.append((slot, ask, float(pred[slot]), edge, book_spread, score))

            candidates.sort(key=lambda c: -c[5])    # by descending score
            print(f"[{now_str()}]   {slug}: {len(candidates)} candidate fire(s) "
                  f"(rule={self.rule}, threshold {self.threshold:+.3f})")

            for slot, ask, pred_val, edge, book_spread, score in candidates:
                await self._fire_token(game_info, slot, ask, pred_val, edge,
                                       book_spread, score, asks_clean, sizes_clean)
                await asyncio.sleep(INTRA_GAME_FIRE_DELAY_S)

    async def _fire_token(self, game_info, slot, ask, pred_val, edge,
                          book_spread, score, asks_24, sizes_24):
        slug = game_info["slug"]
        token_id = game_info["tokens_24"][slot]
        market_type, market_label, side = _slot_label(slot)
        slot_complement = (slot + 12) % 24
        ask_complement = float(asks_24[slot_complement])
        ask_size = float(sizes_24[slot])
        ask_complement_size = float(sizes_24[slot_complement])

        # Limit price: round up to nearest cent. The `- 1e-9` defuzzes floats
        # so that an ask exactly on a cent (e.g. 0.55, which floats to ~55.00...07
        # when multiplied by 100) doesn't bump up to the next cent.
        price = math.ceil(ask * 100 - 1e-9) / 100.0
        size = max(5, math.ceil(1.00 / price))
        notional = price * size

        ts_placed_iso = _now_utc_iso()
        ts_placed_unix = time.time()

        record = {
            "ts_placed": ts_placed_iso,
            "ts_responded": None,
            "game_slug": slug,
            "game_id": game_info["game_id"],
            "game_start_ts": game_info["start_ts"],
            "fire_offset_s": self.t_offset,
            "slot": slot,
            "side": side,
            "market_type": market_type,
            "market_label": market_label,
            "market_id": int(game_info["market_ids_12"][slot % 12]),
            "token_id": token_id,
            "ask": float(ask),
            "ask_size": ask_size,
            "ask_complement": ask_complement,
            "ask_complement_size": ask_complement_size,
            "predicted_prob": float(pred_val),
            "predicted_prob_clipped": float(np.clip(pred_val, 0.0, 1.0)),
            "edge": edge,
            "book_spread": float(book_spread),
            "rule": self.rule,
            "rule_threshold": self.threshold,
            "score": float(score),
            "order_args": {"price": price, "size": size},
            "notional_usdc": notional,
            "placed": False,
            "dry_run": not self.live,
            "status": None,
            "order_id": None,
            "filled_size": None,
            "filled_notional_usdc": None,
            "ws_events_file": None,
            "outcome": None,
            "outcome_filled_at": None,
            "world_idx": None,
            "realised_pnl": None,
        }

        if not self.live:
            record["status"] = "dry_run_skipped"
            self._append_summary(record)
            print(f"[{now_str()}]   DRY {slug} slot{slot:02d} {market_label}/{side} "
                  f"ask={ask:.3f} pred={pred_val:.3f} edge={edge:+.3f} "
                  f"price={price:.2f} size={size}")
            return

        # Atomic check + reservation against the budget cap.
        async with self._budget_lock:
            if self.total_spent + notional > self.budget:
                record["status"] = "budget_exhausted"
                self._append_summary(record)
                print(f"[{now_str()}] BUDGET EXHAUSTED ({self.total_spent:.2f}/{self.budget:.2f}); "
                      f"skipping {slug} slot{slot:02d}")
                return
            # Tentatively reserve the full notional. We'll true it up to
            # makingAmount after the response (or unwind on kill/error).
            self.total_spent += notional
        reserved = notional

        # Set up per-order WS recorder
        ws_log_name = f"{slug}_{slot:02d}_{int(ts_placed_unix)}.jsonl"
        ws_log_path = WS_EVENTS_DIR / ws_log_name
        recorder = WsRecorder(ws_log_path, f"{slug}_slot{slot:02d}", WS_RECORD_DURATION_S)
        self._recorders.append(recorder)
        record["ws_events_file"] = f"ws_events/{ws_log_name}"

        # Place the FOK. Sign with our v2-aware builder (CLOB v2 migration on
        # 2026-04-27 changed the Order schema; see clob_v2.py). Reuse the
        # SDK's `post_order` for auth headers / HTTP delivery — that part is
        # unchanged. Serialize create+post via a single lock to avoid races
        # in the SDK's internal caches.
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY
        args = OrderArgs(price=price, size=size, side=BUY, token_id=token_id)
        sig_type = self.client.builder.sig_type

        async def _place_once():
            async with self._clob_lock:
                signed = await asyncio.to_thread(
                    build_signed_order_v2,
                    client=self.client,
                    order_args=args,
                    sig_type=sig_type,
                )
                return await asyncio.to_thread(
                    self.client.post_order, signed, OrderType.FOK,
                )

        try:
            try:
                resp = await _place_once()
            except Exception as e1:
                msg = str(e1)
                # Retry once on the transient generic-network case only.
                if "Request exception" in msg and "status_code=None" in msg:
                    await asyncio.sleep(1.0)
                    resp = await _place_once()
                else:
                    raise
        except Exception as e:
            record["status"] = f"error: {e}"
            record["ts_responded"] = _now_utc_iso()
            # On payload-rejected errors (HTTP 400 from CLOB), dump the request
            # body we sent so we can see what the server didn't like. Only the
            # first 8 chars of the signature are kept to avoid filling the log.
            try:
                from py_clob_client.clob_types import OrderType as _OT
                sample = build_signed_order_v2(
                    client=self.client, order_args=args, sig_type=sig_type,
                ).dict()
                if "signature" in sample:
                    sample["signature"] = sample["signature"][:10] + "..."
                record["debug_payload_sample"] = sample
            except Exception:
                pass
            self._append_summary(record)
            # Release the reservation since nothing was committed.
            async with self._budget_lock:
                self.total_spent -= reserved
            print(f"[{now_str()}]   ERR {slug} slot{slot:02d}: {e}")
            if "Invalid order payload" in str(e) or "status_code=400" in str(e):
                print(f"[{now_str()}]      payload sample: {record.get('debug_payload_sample')}")
            recorder.close()
            return

        record["ts_responded"] = _now_utc_iso()
        record["placed"] = True
        record["status"] = (resp.get("status") or "").lower()
        record["order_id"] = resp.get("orderID") or resp.get("order_id") or ""
        taking = resp.get("takingAmount")
        making = resp.get("makingAmount")
        # On a buy FOK, takingAmount = shares received, makingAmount = USDC paid.
        # FOK can fill at better-than-limit prices (price improvement), so
        # takingAmount can exceed our requested size; makingAmount is the
        # authoritative USDC notional spent.
        if taking not in (None, ""):
            try:
                record["filled_size"] = float(taking)
            except (TypeError, ValueError):
                pass

        actual_notional: float | None = None
        if making not in (None, ""):
            try:
                actual_notional = float(making)
            except (TypeError, ValueError):
                pass
        if actual_notional is None and record["status"] != "matched":
            # Killed / unmatched — nothing committed.
            actual_notional = 0.0
        if actual_notional is None and record["filled_size"] is not None:
            # Matched but no makingAmount returned — estimate at limit price.
            actual_notional = record["filled_size"] * price

        if actual_notional is not None:
            record["filled_notional_usdc"] = actual_notional
            # Reconcile reservation -> actual.
            async with self._budget_lock:
                self.total_spent += (actual_notional - reserved)

        self._append_summary(record)
        jlog("fire_placed", **{k: v for k, v in record.items()
                                if k not in ("ws_events_file",)})
        print(f"[{now_str()}]   FIRE {slug} slot{slot:02d} {market_label}/{side} "
              f"ask={ask:.3f} pred={pred_val:.3f} edge={edge:+.3f} "
              f"price={price:.2f} size={size} -> {record['status']}")

        # Let recorder run for the full window (in background; close happens
        # when its deadline passes — pruned by the dispatcher).

    def _append_summary(self, record):
        with open(ORDERS_LOG, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        # Count anything that crossed the edge threshold and got a summary
        # row written, regardless of dry-run / placed / killed / budget.
        # In --live this is the count of orders attempted; in dry-run it's
        # the count of would-be orders.
        self.n_trades_attempted += 1

    # ---- main loop ----

    async def _periodic_discovery(self):
        while True:
            await asyncio.sleep(DISCOVERY_REFRESH_S)
            await self.discover_and_schedule()

    def _format_eta(self, secs: float) -> str:
        secs = max(0, int(secs))
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h{m:02d}m"
        if m:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    async def _heartbeat(self):
        """Print one-line status every HEARTBEAT_INTERVAL_S so the operator
        can tell the bot is alive between discoveries and fires."""
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)
            now = time.time()
            pending = [(slug, ts) for slug, ts in self._fire_at.items()
                       if slug not in self._kickoff_slugs and ts > now]
            pending.sort(key=lambda x: x[1])
            if pending:
                next_slug, next_ts = pending[0]
                next_str = f"next: {next_slug} in {self._format_eta(next_ts - now)}"
            else:
                next_str = "no upcoming fires"
            mode = "live" if self.live else "dry"
            print(f"[{now_str()}] heartbeat: scheduled={len(pending)}  "
                  f"kickoffs={len(self._kickoff_slugs)}  "
                  f"trades({mode})={self.n_trades_attempted}  "
                  f"spent=${self.total_spent:.2f}/{self.budget:.2f}  "
                  f"{next_str}")

    async def run(self):
        print(f"[{now_str()}] PregamePCABot starting "
              f"(live={self.live}, budget=${self.budget}, "
              f"rule={self.rule}, threshold={self.threshold}, "
              f"t_offset={self.t_offset}s, model={self.args.model})")

        await self.discover_and_schedule()
        asyncio.create_task(self._periodic_discovery())
        asyncio.create_task(self._heartbeat())

        # Block forever (or until Ctrl-C)
        try:
            while True:
                await asyncio.sleep(60)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass


# ---- main --------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="Actually place orders. Default is dry-run.")
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET,
                    help=f"USDC spend cap (default {DEFAULT_BUDGET})")
    ap.add_argument("--rule", type=str, default=DEFAULT_RULE,
                    choices=["edge", "ratio"],
                    help="Firing rule: 'edge' (fire when predicted_prob−ask > threshold) "
                         "or 'ratio' (fire when edge/max(book_spread, 0.005) > threshold). "
                         f"Default {DEFAULT_RULE}.")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Threshold for the firing rule (default {DEFAULT_THRESHOLD}; "
                         "interpreted as edge for --rule=edge, or as ratio for --rule=ratio)")
    ap.add_argument("--time-seconds", type=int, default=DEFAULT_T_OFFSET_S,
                    help=f"Fire offset relative to game_start (default {DEFAULT_T_OFFSET_S} = -25min)")
    ap.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                    help="Path to .npz model parameters")
    ap.add_argument("--hours-ahead", type=int, default=DEFAULT_HOURS_AHEAD,
                    help=f"Discovery horizon in hours (default {DEFAULT_HOURS_AHEAD})")
    ap.add_argument("--markets", type=str, default="moneyline,totals",
                    help="Comma-separated market types to fire on (default 'moneyline,totals')")
    return ap.parse_args()


def main():
    args = parse_args()
    bot = PregamePCABot(args)
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
