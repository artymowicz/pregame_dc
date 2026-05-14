"""K-redundant wrapper around MarketTracker + MarketWebSocket.

Runs K parallel WS connections, each feeding its own MarketTracker. Reads
go through a single primary tracker; on disconnect the primary fails over
to a healthy peer and the recovering tracker is seeded from the peer's
book state inside `on_connect` (before the WS dispatch loop resumes).

Designed to mitigate the partial-book divergence problem documented in
scratch/reconstruction_compare.py: after a Polymarket WS reconnect the
single-tracker live bot's reset() wipes book state and Polymarket often
fails to re-send book snapshots, leaving the bot with non-sentinel but
wrong best_bid/ask values for many minutes. Redundancy closes that gap
during normal operation; an all-K-down watchdog handles the remaining
case (silent-dead WS, simultaneous disconnect) by triggering a full
resubscribe, which forces fresh book events.
"""
from __future__ import annotations

import asyncio
import time

from pregame_pca.polymarket.sdk import (
    MarketTracker,
    MarketWebSocket,
    log_arb,
    now_str,
)


class RedundantMarketTracker:
    def __init__(
        self,
        token_to_key: dict,
        all_token_ids: list,
        k: int = 2,
        stagger_delay_s: float = 0.0,
        watchdog_interval_s: float = 5.0,
        silent_dead_threshold_s: float = 60.0,
    ):
        # WARNING: stagger_delay_s > 0 causes persistent state drift after a
        # full resubscribe. With a stagger, slot 0 connects first and receives
        # events for `stagger_delay_s` seconds before slot 1; PM does not
        # replay those events to slot 1 when it later subscribes, so slot 1's
        # incoming stream is missing them. The cross-slot disagreement is
        # measurable (~2% of markets persistently disagree at stagger=5s vs
        # 0% at stagger=0). Keep at 0 unless you have a reason — empirically,
        # K=2 simultaneous subscribes do not appear to be rate-limited by PM.
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._k = k
        self._stagger_delay_s = stagger_delay_s
        self._watchdog_interval_s = watchdog_interval_s
        self._silent_dead_threshold_s = silent_dead_threshold_s

        self._token_to_key = token_to_key
        self._all_token_ids = list(all_token_ids)

        self._slots: list[dict] = [
            {
                "ws": None,
                "task": None,
                "tracker": MarketTracker(token_to_key, self._all_token_ids),
                "last_event_ts": 0.0,
            }
            for _ in range(k)
        ]
        self._primary_idx: int = 0
        self._watchdog_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

    # ---- public lifecycle -------------------------------------------------

    async def start(self) -> None:
        for i in range(self._k):
            await self._build_slot(i)
            if i < self._k - 1:
                await asyncio.sleep(self._stagger_delay_s)
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._watchdog_task is not None and not self._watchdog_task.done():
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except (asyncio.CancelledError, Exception):
                pass
            self._watchdog_task = None
        for slot in self._slots:
            await self._stop_slot(slot)

    async def update_token_to_key(self, new_mapping: dict) -> None:
        """Fan out a token_to_key update to every slot (subset short-circuit
        in the bot's discovery refresh path)."""
        for slot in self._slots:
            slot["tracker"].token_to_key.update(new_mapping)

    # ---- drop-in MarketTracker API ---------------------------------------

    @property
    def books(self):
        return self._slots[self._primary_idx]["tracker"].books

    @property
    def token_to_key(self):
        return self._slots[self._primary_idx]["tracker"].token_to_key

    @property
    def dirty_keys(self):
        return self._slots[self._primary_idx]["tracker"].dirty_keys

    def get_best_bid(self, market_id, outcome) -> float:
        return self._resolve_best(market_id, outcome, "bid")[0]

    def get_best_ask(self, market_id, outcome) -> float:
        return self._resolve_best(market_id, outcome, "ask")[0]

    def get_best_bid_with_size(self, market_id, outcome):
        return self._resolve_best(market_id, outcome, "bid")

    def get_best_ask_with_size(self, market_id, outcome):
        return self._resolve_best(market_id, outcome, "ask")

    def reset(self) -> None:
        for slot in self._slots:
            slot["tracker"].reset()

    def consume_dirty_keys(self) -> set:
        out = set()
        for slot in self._slots:
            out |= slot["tracker"].consume_dirty_keys()
        return out

    # ---- internals --------------------------------------------------------

    def _resolve_best(self, market_id, outcome, side: str):
        """Read the best (price, size) from each slot's freshest level.

        Compares prices across slots. On disagreement, picks the slot whose
        best level has the newest server `ts` and warns. Returns sentinel
        values (0.0/0.0 for bid, 1.0/0.0 for ask) if no slot has a level.
        """
        method = (
            "get_best_bid_with_size_and_ts" if side == "bid"
            else "get_best_ask_with_size_and_ts"
        )
        # candidates: list of (slot_idx, price, size, ts)
        candidates: list[tuple[int, float, float, float]] = []
        for j, slot in enumerate(self._slots):
            lvl = getattr(slot["tracker"], method)(market_id, outcome)
            if lvl is not None:
                p, s, ts = lvl
                candidates.append((j, p, s, ts))

        if not candidates:
            return (0.0, 0.0) if side == "bid" else (1.0, 0.0)

        # Pick freshest by per-level ts (ties broken by slot index)
        best_idx, best_p, best_s, best_ts = max(
            candidates, key=lambda c: (c[3], -c[0])
        )

        # Warn if any slot reports a different best price
        prices = {p for _, p, _, _ in candidates}
        if len(prices) > 1:
            items = " ".join(
                f"slot[{j}]=p{p:.4f}/s{s:.2f}@ts{ts:.3f}"
                for j, p, s, ts in candidates
            )
            log_arb(
                f"REDUNDANT_DISAGREE get_best_{side} mid={market_id} "
                f"out={outcome} {items} -> picked slot[{best_idx}]"
                f"=p{best_p:.4f} (freshest ts={best_ts:.3f})",
                quiet=True,
            )
        return (best_p, best_s)

    def _healthy_peer(self, self_idx: int) -> int | None:
        """Return the slot index (!= self_idx) whose tracker has the most
        populated books; None if no peer is non-empty."""
        best_idx = None
        best_n = 0
        for j, slot in enumerate(self._slots):
            if j == self_idx:
                continue
            n = len(slot["tracker"].books)
            if n > best_n:
                best_n = n
                best_idx = j
        return best_idx

    def _make_on_update(self, i: int):
        slot = self._slots[i]
        tracker = slot["tracker"]

        def _on_update(evt: dict) -> None:
            slot["last_event_ts"] = time.time()
            ev_type = evt.get("event_type") or evt.get("type", "")
            if ev_type == "book":
                tracker.process_book(evt)
            elif ev_type == "price_change":
                tracker.process_price_change(evt)
        return _on_update

    def _make_on_connect(self, i: int):
        async def _on_connect() -> None:
            self_tracker = self._slots[i]["tracker"]
            # Failover BEFORE reset+seed so the seed window is invisible to
            # readers (the freshly-reset slot is briefly empty between reset
            # and the copy loop, even though no `await` runs in between).
            if i == self._primary_idx:
                new_primary = self._healthy_peer(i)
                if new_primary is not None:
                    self._primary_idx = new_primary
                    log_arb(
                        f"REDUNDANT_FAILOVER primary {i} -> {new_primary} "
                        f"(slot {i} reconnecting)",
                        quiet=True,
                    )
            self_tracker.reset()
            peer_idx = self._healthy_peer(i)
            if peer_idx is not None:
                # Single synchronous block — no await between mutations, so
                # the peer cannot run code that mutates its books in here.
                peer = self._slots[peer_idx]["tracker"]
                for key, bs in peer.books.items():
                    self_tracker.books[key] = bs.per_game_data()
                self_tracker.most_recent_ts = peer.most_recent_ts
                log_arb(
                    f"REDUNDANT_SEED slot {i} seeded from slot {peer_idx} "
                    f"({len(self_tracker.books)} keys)",
                    quiet=True,
                )
            else:
                log_arb(
                    f"REDUNDANT_SEED slot {i} reset with no healthy peer "
                    f"(initial connect or all-down)",
                    quiet=True,
                )
        return _on_connect

    async def _build_slot(self, i: int) -> None:
        slot = self._slots[i]
        await self._stop_slot(slot)
        ws = MarketWebSocket(token_ids=self._all_token_ids)
        ws.on_update = self._make_on_update(i)
        ws.on_connect = self._make_on_connect(i)
        slot["ws"] = ws
        slot["task"] = asyncio.create_task(ws.run())
        log_arb(
            f"REDUNDANT_BUILD slot {i} started "
            f"({len(self._all_token_ids)} tokens)",
            quiet=True,
        )

    async def _stop_slot(self, slot: dict) -> None:
        ws = slot["ws"]
        task = slot["task"]
        if ws is None:
            return
        ws.stop()
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                task.cancel()
        slot["ws"] = None
        slot["task"] = None

    async def _full_resubscribe(self) -> None:
        log_arb(f"REDUNDANT_RESUB tearing down all {self._k} slots", quiet=False)
        for slot in self._slots:
            await self._stop_slot(slot)
            slot["last_event_ts"] = 0.0
        # Resetting trackers wipes the books that fed _healthy_peer; after
        # resubscribe each slot will receive fresh book events.
        for slot in self._slots:
            slot["tracker"].reset()
        for i in range(self._k):
            await self._build_slot(i)
            if i < self._k - 1:
                await asyncio.sleep(self._stagger_delay_s)
        log_arb(f"REDUNDANT_RESUB rebuild complete", quiet=False)

    async def _watchdog_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._watchdog_interval_s,
                )
                return  # stop_event set
            except asyncio.TimeoutError:
                pass

            now = time.time()
            any_warm = any(s["last_event_ts"] > 0 for s in self._slots)
            silent = (
                any_warm
                and all(
                    now - s["last_event_ts"] > self._silent_dead_threshold_s
                    for s in self._slots
                )
            )
            all_disconnected = all(
                s["ws"] is not None and not s["ws"].is_connected
                for s in self._slots
            )

            if silent or all_disconnected:
                reason = []
                if silent:
                    reason.append(
                        f"silent>{self._silent_dead_threshold_s:.0f}s"
                    )
                if all_disconnected:
                    reason.append("all_disconnected")
                log_arb(
                    f"REDUNDANT_WATCHDOG triggering full resubscribe "
                    f"({', '.join(reason)})",
                    quiet=False,
                )
                async with self._lock:
                    try:
                        await self._full_resubscribe()
                    except Exception as e:
                        log_arb(
                            f"REDUNDANT_RESUB failed: {e}",
                            quiet=False,
                        )
