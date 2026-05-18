"""Backtest replay with PnL: SC and TX vs live, on the 45 overlap games.

For each of the 45 games in {live matched} ∩ {SC} ∩ {TX}, runs the model
at the live fire time using each backtest source, identifies the slots
that would fire (edge > 0.04 in moneyline + totals), and reports fill
PnL using the slot outcomes resolved from Gamma.

Fetches any missing outcomes inline (Gamma is anonymous, fast).
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from pregame_dc import paths
from pregame_dc.constants import TYPE_FOR_SLOT, MARKET_LABELS
from pregame_dc.pipelines.helpers import canonical_slot_market_ids

LIVE_RESOLVED = paths.PACKAGE_ROOT / "pregame_dc" / "live" / "logs" / "orders_summary_resolved.jsonl"
SC_DIR = paths.PACKAGE_ROOT / "data" / "self_collected" / "per_game_data"
TX_DIR = paths.PACKAGE_ROOT / "data" / "telonex" / "per_game_data"

THRESHOLD = 0.04
ASK_LO, ASK_HI = 0.01, 0.99
ENABLED_MARKETS = {"moneyline", "totals"}
ENABLED_24 = sorted(
    [s for s, mt in TYPE_FOR_SLOT.items() if mt in ENABLED_MARKETS]
    + [s + 12 for s, mt in TYPE_FOR_SLOT.items() if mt in ENABLED_MARKETS]
)

MODEL_BY_OFFSET = {-1500: paths.MODEL_T_25MIN, -600: paths.MODEL_T_10MIN}

# Bot's `--markets` config changed mid-run. All matched totals fires
# happened on 2026-05-03 between 15:05 and 23:05 UTC; from May 4 the bot
# was started with `--markets moneyline` only. To mirror what the bot
# would actually fire on per game, restrict the backtest's candidate
# slots to (moneyline + totals) for games before this cutoff, and
# (moneyline only) after.
CONFIG_CUTOFF_TS = 1778025600   # 2026-05-04 00:00 UTC
SLOTS_BY_ENABLED = {
    frozenset({"moneyline"}): sorted(
        [s for s, mt in TYPE_FOR_SLOT.items() if mt == "moneyline"]
        + [s + 12 for s, mt in TYPE_FOR_SLOT.items() if mt == "moneyline"]),
    frozenset({"moneyline", "totals"}): sorted(
        [s for s, mt in TYPE_FOR_SLOT.items() if mt in {"moneyline", "totals"}]
        + [s + 12 for s, mt in TYPE_FOR_SLOT.items() if mt in {"moneyline", "totals"}]),
}


def enabled_slots_for_game(game_start_ts: int) -> list[int]:
    """Mirror the bot's `--markets` config at the time this game fired."""
    if game_start_ts < CONFIG_CUTOFF_TS:
        return SLOTS_BY_ENABLED[frozenset({"moneyline", "totals"})]
    return SLOTS_BY_ENABLED[frozenset({"moneyline"})]

GAMMA_URL = "https://gamma-api.polymarket.com/markets/{id}"


def load_model(npz_path):
    d = np.load(npz_path)
    mu, sd, beta_K = d["mu"], d["sd_safe"], d["beta_K"]
    def predict(x_24):
        z = (x_24 - mu) / sd
        F = np.concatenate([z, [1.0]])
        return beta_K @ F
    return predict


def build_tx_index():
    idx = {}
    for p in TX_DIR.glob("*.parquet"):
        m = re.match(r"^(.+)-\d{4}-\d{2}-\d{2}$", p.stem)
        idx[m.group(1) if m else p.stem] = p
    return idx


def ask_at_t(df, market_id, is_yes, t_ms):
    sub = df[(df["market_id"] == market_id) & (df["is_yes"] == is_yes)
             & (df["timestamp_ms"] <= t_ms)]
    if sub.empty:
        return 1.0
    row = sub.iloc[sub["timestamp_ms"].values.argmax()]
    return float(row["best_ask"])


def vector_24(df, slot_mids, t_ms):
    x = np.full(24, 1.0)
    for s, mid in enumerate(slot_mids):
        x[s] = ask_at_t(df, mid, True, t_ms)
        x[s + 12] = ask_at_t(df, mid, False, t_ms)
    return x


def derive_slot_mids(df):
    unique = df.groupby("market_id").first().reset_index()
    return canonical_slot_market_ids(unique.to_dict("records"))


def fetch_outcome(session, market_id):
    for attempt in range(4):
        try:
            r = session.get(GAMMA_URL.format(id=market_id), timeout=20)
            if r.status_code == 404:
                return (market_id, None)
            r.raise_for_status()
            m = r.json()
            prices = json.loads(m.get("outcomePrices", "[]"))
            yes_price = float(prices[0]) if prices else None
            closed = bool(m.get("closed", False))
            return (market_id, yes_price if closed else None)
        except Exception:
            time.sleep(1.5 ** attempt)
    return (market_id, None)


def main():
    # ---- load live + build overlap set ----
    all_attempts = []
    with LIVE_RESOLVED.open() as f:
        for line in f:
            if line.strip():
                all_attempts.append(json.loads(line))

    # Bot's intent-to-fire = any logged attempt (matched + errored + dry_run).
    # 'fired_slugs' must restrict to games where at least one attempt
    # MATCHED so we know the game window aligned with available data, but
    # the LIVE fire SET counted below is all attempts for those games.
    matched_slugs = sorted({r["game_slug"] for r in all_attempts if r["status"] == "matched"})
    tx_index = build_tx_index()
    sc_slugs = {p.stem for p in SC_DIR.glob("*.parquet")}
    overlap = sorted(set(matched_slugs) & sc_slugs & set(tx_index))
    print(f"45-game overlap (live matched ∩ SC ∩ TX): {len(overlap)}")

    fire_off_by_slug = {r["game_slug"]: r["fire_offset_s"] for r in all_attempts}
    fire_start_by_slug = {r["game_slug"]: r["game_start_ts"] for r in all_attempts}
    # Live log stores `slot` as the 0..23 index (e.g. 14 == NO side of
    # canonical slot 2). Key by that directly.
    live_by_game_slot = {(r["game_slug"], r["slot"]): r for r in all_attempts}

    # ---- derive slot mappings per source (SC and TX use different
    #      market_id namespaces but condition_ids match). Outcomes are
    #      looked up via SC's Gamma-compatible market_ids.
    print("\nderiving slot mappings & collecting market_ids ...")
    sc_mids_by_slug = {}
    tx_mids_by_slug = {}
    slug_to_sc_df = {}
    slug_to_tx_df = {}
    needed_mids = set()   # SC (Gamma) ids only — these go to outcomes
    for slug in overlap:
        sc_df = pq.read_table(SC_DIR / f"{slug}.parquet").to_pandas()
        tx_df = pq.read_table(tx_index[slug]).to_pandas()
        sc_mids = derive_slot_mids(sc_df)
        tx_mids = derive_slot_mids(tx_df)
        if sc_mids is None or tx_mids is None:
            print(f"  WARN: {slug} missing canonical 12-slot set "
                  f"(sc={sc_mids is not None}, tx={tx_mids is not None})")
            continue
        sc_mids_by_slug[slug] = sc_mids
        tx_mids_by_slug[slug] = tx_mids
        slug_to_sc_df[slug] = sc_df
        slug_to_tx_df[slug] = tx_df
        needed_mids.update(sc_mids)
    print(f"  {len(sc_mids_by_slug)} games with full 12-slot mapping in both sources")
    print(f"  unique SC (Gamma) market_ids needed for outcomes: {len(needed_mids)}")

    # ---- merge with outcomes.parquet, fetch missing inline ----
    out_df = pq.read_table(paths.OUTCOMES).to_pandas()
    outcomes = {int(r.market_id): r.final_price
                for r in out_df.itertuples(index=False)
                if not pd.isna(r.final_price)}
    have = needed_mids & set(outcomes)
    missing = sorted(needed_mids - set(outcomes))
    print(f"\noutcomes already cached: {len(have)} / {len(needed_mids)}")
    print(f"to fetch from Gamma:     {len(missing)}")

    if missing:
        session = requests.Session()
        session.headers["User-Agent"] = "pregame_dc-backtest-subset/1.0"
        t0 = time.time()
        n_fetched = 0
        with ThreadPoolExecutor(max_workers=20) as ex:
            futures = [ex.submit(fetch_outcome, session, mid) for mid in missing]
            for fut in as_completed(futures):
                mid, fp = fut.result()
                if fp is not None:
                    outcomes[mid] = fp
                n_fetched += 1
        print(f"  fetched {n_fetched} in {time.time()-t0:.1f}s "
              f"({sum(1 for m in missing if m in outcomes)} resolved)")

    # ---- replay backtest per game per source ----
    models = {off: load_model(p) for off, p in MODEL_BY_OFFSET.items()}

    # Tally:
    #   per source: total fires, total notional, total pnl, wins/losses
    summary = {"SC": {}, "TX": {}, "LIVE": {}}
    for s in summary:
        summary[s] = {"fires": 0, "notional": 0.0, "pnl": 0.0, "wins": 0,
                      "losses": 0, "unresolved": 0}

    per_game_records = []
    for slug in overlap:
        if slug not in sc_mids_by_slug:
            continue
        fire_off = fire_off_by_slug[slug]
        fire_time_ms = (fire_start_by_slug[slug] + fire_off) * 1000
        sc_mids = sc_mids_by_slug[slug]
        tx_mids = tx_mids_by_slug[slug]
        predict = models[fire_off]

        sc_x = vector_24(slug_to_sc_df[slug], sc_mids, fire_time_ms)
        tx_x = vector_24(slug_to_tx_df[slug], tx_mids, fire_time_ms)
        sc_pred = predict(sc_x)
        tx_pred = predict(tx_x)

        for slot_24 in enabled_slots_for_game(fire_start_by_slug[slug]):
            slot_canon = slot_24 if slot_24 < 12 else slot_24 - 12
            is_yes = slot_24 < 12
            side = "Y" if is_yes else "N"
            # Outcomes are indexed by SC's Gamma market_id (the same id used
            # in outcomes.parquet). The TX market_id at the same canonical
            # slot resolves to the same underlying outcome.
            mid = sc_mids[slot_canon]
            y_yes = outcomes.get(int(mid))   # YES-side outcome, 0/1 or None
            outcome_side = (y_yes if is_yes
                            else (None if y_yes is None else 1.0 - y_yes))

            # SC backtest fire?
            sc_ask = float(sc_x[slot_24])
            sc_edge = float(sc_pred[slot_24] - sc_ask)
            if ASK_LO <= sc_ask <= ASK_HI and sc_edge > THRESHOLD:
                summary["SC"]["fires"] += 1
                summary["SC"]["notional"] += sc_ask
                if outcome_side is None:
                    summary["SC"]["unresolved"] += 1
                else:
                    pnl = outcome_side - sc_ask
                    summary["SC"]["pnl"] += pnl
                    if outcome_side == 1: summary["SC"]["wins"] += 1
                    else: summary["SC"]["losses"] += 1

            # TX
            tx_ask = float(tx_x[slot_24])
            tx_edge = float(tx_pred[slot_24] - tx_ask)
            if ASK_LO <= tx_ask <= ASK_HI and tx_edge > THRESHOLD:
                summary["TX"]["fires"] += 1
                summary["TX"]["notional"] += tx_ask
                if outcome_side is None:
                    summary["TX"]["unresolved"] += 1
                else:
                    pnl = outcome_side - tx_ask
                    summary["TX"]["pnl"] += pnl
                    if outcome_side == 1: summary["TX"]["wins"] += 1
                    else: summary["TX"]["losses"] += 1

            # LIVE: any logged attempt (matched + errored) — the bot
            # decided to fire. PnL is computed hypothetically using the
            # recorded ask and the underlying slot outcome (which is
            # independent of whether Polymarket accepted the order).
            live = live_by_game_slot.get((slug, slot_24))
            if live is not None:
                summary["LIVE"]["fires"] += 1
                summary["LIVE"]["notional"] += live["ask"]
                if outcome_side is None:
                    summary["LIVE"]["unresolved"] += 1
                else:
                    pnl = outcome_side - live["ask"]
                    summary["LIVE"]["pnl"] += pnl
                    if outcome_side == 1: summary["LIVE"]["wins"] += 1
                    else: summary["LIVE"]["losses"] += 1
                # Bookkeeping for matched vs all
                if live["status"] == "matched":
                    summary["LIVE"].setdefault("matched_fires", 0)
                    summary["LIVE"]["matched_fires"] += 1

    # ---- report ----
    print(f"\n{'='*70}\nBacktest replay PnL on the 45 overlap games "
          f"(threshold={THRESHOLD}, markets={'+'.join(ENABLED_MARKETS)}, share=1.0)")
    print(f"{'='*70}")
    print(f"{'source':<6s}  {'fires':>6s}  {'wins':>5s}  {'losses':>7s}  "
          f"{'win%':>6s}  {'PnL/share':>10s}  {'total PnL':>10s}  {'unresolved':>10s}")
    for src in ("LIVE", "SC", "TX"):
        d = summary[src]
        resolved = d["wins"] + d["losses"]
        win_pct = d["wins"] / resolved * 100 if resolved else 0
        avg = d["pnl"] / resolved if resolved else 0
        print(f"{src:<6s}  {d['fires']:>6d}  {d['wins']:>5d}  {d['losses']:>7d}  "
              f"{win_pct:>5.1f}%  {avg:>+10.4f}  {d['pnl']:>+10.4f}  {d['unresolved']:>10d}")

    matched_live = summary["LIVE"].get("matched_fires", 0)
    print(f"\nLIVE fires = all attempts (bot's intent to fire). "
          f"Of {summary['LIVE']['fires']} live attempts: "
          f"{matched_live} matched, {summary['LIVE']['fires'] - matched_live} errored.")
    print(f"'PnL/share' is per unit notional (outcome − ask); for real PnL "
          f"multiply by the bot's $5 (or sized) bet per fire.")

    # ---- Decompose: why did the bot not attempt the cells the backtest fires on? ----
    print(f"\n{'='*70}")
    print(f"Disagreement decomposition between backtest fire set and LIVE fire set")
    print(f"  (per-game --markets config: moneyline+totals before "
          f"{CONFIG_CUTOFF_TS}, moneyline only after)")
    print(f"{'='*70}")
    n_no_live_data = 0
    n_live_ask_placeholder = 0
    n_live_ask_out_of_range = 0
    n_live_edge_below_thr = 0
    n_live_attempted = 0
    n_btx_both_fired = 0
    other = 0
    # Symmetric: cells where live fired but backtest didn't
    n_live_only = 0
    n_live_only_sc_no = 0
    n_live_only_tx_no = 0

    # Need to re-scan and parse live's recorded 24-vector for each game
    from pregame_dc.analyze.full_24vec_compare import (
        parse_live_24_vectors, _detect_bot_tz_offset_ms
    )
    BOT_STDOUT = paths.PACKAGE_ROOT / "pregame_dc" / "live" / "logs" / "bot_stdout.log"
    tz_off = _detect_bot_tz_offset_ms(BOT_STDOUT, [r for r in all_attempts if r["status"] == "matched"])
    live_vecs = parse_live_24_vectors(BOT_STDOUT, tz_offset_ms=tz_off)

    examples = []   # (slug, slot_canon, side, reason, ...)
    for slug in overlap:
        if slug not in sc_mids_by_slug:
            continue
        fire_off = fire_off_by_slug[slug]
        fire_time_ms = (fire_start_by_slug[slug] + fire_off) * 1000
        sc_mids = sc_mids_by_slug[slug]
        tx_mids = tx_mids_by_slug[slug]
        sc_x = vector_24(slug_to_sc_df[slug], sc_mids, fire_time_ms)
        tx_x = vector_24(slug_to_tx_df[slug], tx_mids, fire_time_ms)
        sc_pred = models[fire_off](sc_x)
        tx_pred = models[fire_off](tx_x)

        for slot_24 in enabled_slots_for_game(fire_start_by_slug[slug]):
            slot_canon = slot_24 if slot_24 < 12 else slot_24 - 12
            side = "Y" if slot_24 < 12 else "N"
            sc_ask = float(sc_x[slot_24])
            tx_ask = float(tx_x[slot_24])
            sc_edge = float(sc_pred[slot_24] - sc_ask)
            tx_edge = float(tx_pred[slot_24] - tx_ask)
            sc_fired = ASK_LO <= sc_ask <= ASK_HI and sc_edge > THRESHOLD
            tx_fired = ASK_LO <= tx_ask <= ASK_HI and tx_edge > THRESHOLD
            live = live_by_game_slot.get((slug, slot_24))

            # Symmetric: LIVE fired but backtest didn't
            if live is not None and not (sc_fired and tx_fired):
                n_live_only += 1
                if not sc_fired: n_live_only_sc_no += 1
                if not tx_fired: n_live_only_tx_no += 1
                continue

            if not (sc_fired and tx_fired):
                continue

            # Backtest fires here
            n_btx_both_fired += 1
            if live is not None:
                n_live_attempted += 1
                continue

            # Backtest fires but live didn't attempt. Inspect live's vector.
            lv = live_vecs.get(slug)
            if lv is None:
                n_no_live_data += 1
                continue
            # Re-run the model on the live 24-vector (more faithful than
            # approximating NO pred as 1 - YES pred).
            live_full_pred = models[fire_off](lv["x"])
            live_ask = float(lv["x"][slot_24])
            live_pred = float(live_full_pred[slot_24])
            live_edge = live_pred - live_ask
            if live_ask >= 1.0:
                n_live_ask_placeholder += 1
                reason = "live_ask=1.0 (placeholder)"
            elif not (ASK_LO <= live_ask <= ASK_HI):
                n_live_ask_out_of_range += 1
                reason = f"live_ask={live_ask:.3f} out of [{ASK_LO},{ASK_HI}]"
            elif live_edge <= THRESHOLD:
                n_live_edge_below_thr += 1
                reason = (f"live_edge={live_edge:+.4f} ≤ {THRESHOLD} "
                          f"(live_ask={live_ask:.3f} vs sc_ask={sc_ask:.3f} tx_ask={tx_ask:.3f})")
            else:
                other += 1
                reason = "other"
            if len(examples) < 10:
                examples.append((slug, slot_canon, side, reason,
                                 live_ask, sc_ask, tx_ask, live_edge, sc_edge, tx_edge))

    print(f"Backtest fires (SC ∩ TX) on enabled markets per game: n={n_btx_both_fired}")
    print(f"  live ATTEMPTED:                              {n_live_attempted}")
    print(f"  live SKIPPED — no live stdout for game:      {n_no_live_data}")
    print(f"  live SKIPPED — live_ask = 1.0 (no quote):    {n_live_ask_placeholder}")
    print(f"  live SKIPPED — live_ask out of [0.01,0.99]:  {n_live_ask_out_of_range}")
    print(f"  live SKIPPED — live edge ≤ {THRESHOLD}:           {n_live_edge_below_thr}")
    print(f"  live SKIPPED — other (unexplained):          {other}")
    print(f"\nLIVE fires that backtest (SC ∩ TX) did NOT fire on: n={n_live_only}")
    print(f"  (SC didn't fire: {n_live_only_sc_no}, TX didn't fire: {n_live_only_tx_no})")

    # ---- LIVE PnL decomposed by fire-set membership ----
    print(f"\n{'='*70}")
    print(f"LIVE PnL decomposed by fire-set membership")
    print(f"{'='*70}")
    live_inter_pnl = live_inter_n = 0
    live_only_pnl = live_only_n = 0
    for slug in overlap:
        if slug not in sc_mids_by_slug:
            continue
        fire_off = fire_off_by_slug[slug]
        fire_time_ms = (fire_start_by_slug[slug] + fire_off) * 1000
        sc_mids = sc_mids_by_slug[slug]
        tx_mids = tx_mids_by_slug[slug]
        sc_x = vector_24(slug_to_sc_df[slug], sc_mids, fire_time_ms)
        tx_x = vector_24(slug_to_tx_df[slug], tx_mids, fire_time_ms)
        sc_pred = models[fire_off](sc_x)
        tx_pred = models[fire_off](tx_x)
        for slot_24 in enabled_slots_for_game(fire_start_by_slug[slug]):
            slot_canon = slot_24 if slot_24 < 12 else slot_24 - 12
            is_yes = slot_24 < 12
            live = live_by_game_slot.get((slug, slot_24))
            if live is None:
                continue
            mid = sc_mids[slot_canon]
            y_yes = outcomes.get(int(mid))
            outcome_side = (y_yes if is_yes else (1.0 - y_yes if y_yes is not None else None))
            if outcome_side is None:
                continue
            sc_ask = float(sc_x[slot_24]); sc_edge = float(sc_pred[slot_24] - sc_ask)
            tx_ask = float(tx_x[slot_24]); tx_edge = float(tx_pred[slot_24] - tx_ask)
            sc_fired = ASK_LO <= sc_ask <= ASK_HI and sc_edge > THRESHOLD
            tx_fired = ASK_LO <= tx_ask <= ASK_HI and tx_edge > THRESHOLD
            pnl = outcome_side - live["ask"]
            if sc_fired and tx_fired:
                live_inter_pnl += pnl; live_inter_n += 1
            else:
                live_only_pnl += pnl; live_only_n += 1
    print(f"  LIVE ∩ backtest (SC ∩ TX) fires:  n={live_inter_n}  "
          f"PnL=${live_inter_pnl:+.4f}  avg=${live_inter_pnl/live_inter_n if live_inter_n else 0:+.4f}")
    print(f"  LIVE-only (backtest skipped):     n={live_only_n}  "
          f"PnL=${live_only_pnl:+.4f}  avg=${live_only_pnl/live_only_n if live_only_n else 0:+.4f}")

    print(f"\nExamples of cells where live edge was below threshold while backtest fired:")
    print(f"  {'slug':<22s} {'slot/side':<10s} {'live ask':>10s} {'sc ask':>8s} {'tx ask':>8s} "
          f"{'live edge':>10s} {'sc edge':>9s} {'tx edge':>9s}")
    for slug, sc_, sd, reason, lask, scask, txask, ledge, scedge, txedge in examples:
        print(f"  {slug:<22s} {sc_:>3d}/{sd}     {lask:>10.4f} {scask:>8.4f} {txask:>8.4f} "
              f"{ledge:>+10.4f} {scedge:>+9.4f} {txedge:>+9.4f}")


if __name__ == "__main__":
    main()
