"""Backtest replay on the full 388-game SC ∩ TX ∩ LIVE overlap.

Treats every live attempt (matched OR errored, most commonly 403 geo-block)
as a valid hypothetical fill at the recorded ask. Games where the live
bot evaluated but its edge calc didn't cross threshold contribute 0 LIVE
fires for those slots.

Per-game --markets config and fire_offset are inferred from a single
cutoff (the May-4-01:12-UTC restart that changed both):
  * before: t_offset = -1500s, --markets moneyline,totals
  * after:  t_offset = -600s,  --markets moneyline
"""
from __future__ import annotations

import argparse
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
BOT_STDOUT = paths.PACKAGE_ROOT / "pregame_dc" / "live" / "logs" / "bot_stdout.log"
SC_DIR = paths.PACKAGE_ROOT / "data" / "self_collected" / "per_game_data"
TX_DIR = paths.PACKAGE_ROOT / "data" / "telonex" / "per_game_data"
OUT_CSV = paths.PACKAGE_ROOT / "plots" / "backtest_full_overlap.csv"

THRESHOLD = 0.04
ASK_LO, ASK_HI = 0.01, 0.99

MODEL_BY_OFFSET = {-1500: paths.MODEL_T_25MIN, -600: paths.MODEL_T_10MIN}

# Single cutoff for both fire_offset_s and --markets config (bot restart
# at 2026-05-04 01:12 UTC switched t_offset and removed totals together).
CONFIG_CUTOFF_TS = 1778022720   # 2026-05-04 01:12:00 UTC

# Window during which the bot's VPN was down → 403 "Trading restricted"
# errors on every attempt. Empirically derived from earliest/latest 403 in
# orders_summary_resolved.jsonl. Games whose fire time falls in this
# range are excluded entirely.
VPN_DOWN_START_TS = 1778251801   # 2026-05-08 14:50:01 UTC (first 403)
VPN_DOWN_END_TS   = 1778373301   # 2026-05-10 00:35:01 UTC (last 403)


def fire_offset_for(game_start_ts: int) -> int:
    return -1500 if game_start_ts < CONFIG_CUTOFF_TS else -600


def enabled_slots_for(game_start_ts: int) -> list[int]:
    types = ({"moneyline", "totals"} if game_start_ts < CONFIG_CUTOFF_TS
             else {"moneyline"})
    ys = [s for s, mt in TYPE_FOR_SLOT.items() if mt in types]
    return sorted(ys + [s + 12 for s in ys])


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
            r = session.get(f"https://gamma-api.polymarket.com/markets/{market_id}", timeout=20)
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


def parse_live_game_starts():
    """Parse bot_stdout for 'firing for game {slug} (start_ts={N})' lines.
    Returns {slug: game_start_ts}."""
    fire_re = re.compile(r"\] firing for game (\S+) \(start_ts=(\d+)\)")
    out = {}
    with BOT_STDOUT.open() as f:
        for line in f:
            m = fire_re.search(line)
            if m:
                out[m.group(1)] = int(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-vpn-window", action="store_true",
                    help="keep games whose fire time fell in the VPN-down "
                         "window (default: drop them)")
    args = ap.parse_args()

    # ---- universe: SC ∩ TX ∩ LIVE (388 games) ----
    sc_slugs = {p.stem for p in SC_DIR.glob("*.parquet")}
    tx_index = build_tx_index()
    tx_slugs = set(tx_index)
    live_starts = parse_live_game_starts()
    live_slugs = set(live_starts)
    overlap_raw = sorted(sc_slugs & tx_slugs & live_slugs)
    print(f"SC={len(sc_slugs)}, TX={len(tx_slugs)}, LIVE={len(live_slugs)}")
    print(f"3-way overlap (before VPN-down filter): {len(overlap_raw)} games")

    if args.include_vpn_window:
        overlap = list(overlap_raw)
        print(f"VPN-window filter DISABLED — keeping all {len(overlap)} games")
    else:
        def fire_time_ts(slug):
            gst = live_starts[slug]
            return gst + fire_offset_for(gst)
        overlap = [s for s in overlap_raw
                   if not (VPN_DOWN_START_TS <= fire_time_ts(s) <= VPN_DOWN_END_TS)]
        n_dropped = len(overlap_raw) - len(overlap)
        print(f"dropped {n_dropped} games whose fire time fell in the VPN-down window "
              f"({VPN_DOWN_START_TS} → {VPN_DOWN_END_TS})")
        print(f"final overlap: {len(overlap)} games")

    # ---- live attempts (matched + errored + dry_run all count) ----
    attempts = [json.loads(l) for l in LIVE_RESOLVED.open() if l.strip()]
    live_by_game_slot = {(r["game_slug"], r["slot"]): r for r in attempts}
    # Status histogram on the overlap games
    from collections import Counter
    overlap_attempts = [r for r in attempts if r["game_slug"] in set(overlap)]
    print(f"\nlive attempts on overlap games: {len(overlap_attempts)}")
    print(f"  matched: {sum(1 for r in overlap_attempts if r['status'] == 'matched')}")
    print(f"  errored: {sum(1 for r in overlap_attempts if r['status'].startswith('error'))}")
    print(f"  dry_run: {sum(1 for r in overlap_attempts if r['status'] == 'dry_run_skipped')}")
    err_types = Counter()
    for r in overlap_attempts:
        if r["status"].startswith("error"):
            err_types[r["status"][:80]] += 1
    if err_types:
        print(f"  error breakdown:")
        for t, c in err_types.most_common():
            print(f"    {c:>3d}  {t}")

    # ---- derive slot mappings, collect outcome market_ids ----
    print(f"\nderiving slot mappings ...")
    sc_mids_by_slug = {}
    tx_mids_by_slug = {}
    slug_to_sc_df = {}
    slug_to_tx_df = {}
    needed_mids = set()
    n_skipped_no_full_set = 0
    for slug in overlap:
        sc_df = pq.read_table(SC_DIR / f"{slug}.parquet").to_pandas()
        tx_df = pq.read_table(tx_index[slug]).to_pandas()
        sc_mids = derive_slot_mids(sc_df)
        tx_mids = derive_slot_mids(tx_df)
        if sc_mids is None or tx_mids is None:
            n_skipped_no_full_set += 1
            continue
        sc_mids_by_slug[slug] = sc_mids
        tx_mids_by_slug[slug] = tx_mids
        slug_to_sc_df[slug] = sc_df
        slug_to_tx_df[slug] = tx_df
        needed_mids.update(sc_mids)
    print(f"  {len(sc_mids_by_slug)}/{len(overlap)} games with full 12-slot mapping in both sources "
          f"(skipped {n_skipped_no_full_set})")
    print(f"  unique market_ids needed: {len(needed_mids)}")

    # ---- outcomes ----
    out_df = pq.read_table(paths.OUTCOMES).to_pandas()
    outcomes = {int(r.market_id): r.final_price for r in out_df.itertuples(index=False)
                if not pd.isna(r.final_price)}
    have = needed_mids & set(outcomes)
    missing = sorted(needed_mids - set(outcomes))
    print(f"\noutcomes cached: {len(have)}, fetching {len(missing)} ...")
    if missing:
        session = requests.Session()
        session.headers["User-Agent"] = "pregame_dc-backtest-full/1.0"
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=20) as ex:
            futures = [ex.submit(fetch_outcome, session, mid) for mid in missing]
            for fut in as_completed(futures):
                mid, fp = fut.result()
                if fp is not None:
                    outcomes[mid] = fp
        print(f"  done in {time.time()-t0:.1f}s "
              f"({sum(1 for m in missing if m in outcomes)} resolved)")

    # ---- replay ----
    models = {off: load_model(p) for off, p in MODEL_BY_OFFSET.items()}
    summary = {s: {"fires": 0, "notional": 0.0, "pnl": 0.0,
                   "wins": 0, "losses": 0, "unresolved": 0,
                   "matched_fires": 0, "errored_fires": 0}
               for s in ("SC", "TX", "LIVE")}

    # Per-cell records for the output CSV
    records = []

    for slug in overlap:
        if slug not in sc_mids_by_slug:
            continue
        game_start_ts = live_starts[slug]
        fire_off = fire_offset_for(game_start_ts)
        fire_time_ms = (game_start_ts + fire_off) * 1000
        enabled = enabled_slots_for(game_start_ts)
        sc_mids = sc_mids_by_slug[slug]
        tx_mids = tx_mids_by_slug[slug]
        predict = models[fire_off]

        sc_x = vector_24(slug_to_sc_df[slug], sc_mids, fire_time_ms)
        tx_x = vector_24(slug_to_tx_df[slug], tx_mids, fire_time_ms)
        sc_pred = predict(sc_x)
        tx_pred = predict(tx_x)

        for slot_24 in enabled:
            slot_canon = slot_24 if slot_24 < 12 else slot_24 - 12
            is_yes = slot_24 < 12
            side = "Y" if is_yes else "N"
            mid = sc_mids[slot_canon]
            y_yes = outcomes.get(int(mid))
            outcome_side = (y_yes if is_yes
                            else (None if y_yes is None else 1.0 - y_yes))

            # SC
            sc_ask = float(sc_x[slot_24])
            sc_edge = float(sc_pred[slot_24] - sc_ask)
            sc_fired = ASK_LO <= sc_ask <= ASK_HI and sc_edge > THRESHOLD
            if sc_fired:
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
            tx_fired = ASK_LO <= tx_ask <= ASK_HI and tx_edge > THRESHOLD
            if tx_fired:
                summary["TX"]["fires"] += 1
                summary["TX"]["notional"] += tx_ask
                if outcome_side is None:
                    summary["TX"]["unresolved"] += 1
                else:
                    pnl = outcome_side - tx_ask
                    summary["TX"]["pnl"] += pnl
                    if outcome_side == 1: summary["TX"]["wins"] += 1
                    else: summary["TX"]["losses"] += 1

            # LIVE: any logged attempt (matched, errored, dry_run)
            live = live_by_game_slot.get((slug, slot_24))
            if live is not None:
                summary["LIVE"]["fires"] += 1
                summary["LIVE"]["notional"] += live["ask"]
                if live["status"] == "matched":
                    summary["LIVE"]["matched_fires"] += 1
                elif live["status"].startswith("error"):
                    summary["LIVE"]["errored_fires"] += 1
                if outcome_side is None:
                    summary["LIVE"]["unresolved"] += 1
                else:
                    pnl = outcome_side - live["ask"]
                    summary["LIVE"]["pnl"] += pnl
                    if outcome_side == 1: summary["LIVE"]["wins"] += 1
                    else: summary["LIVE"]["losses"] += 1

            # Record cells where at least one source fired (or live attempted)
            if sc_fired or tx_fired or (live is not None):
                records.append({
                    "game_slug": slug,
                    "game_start_ts": game_start_ts,
                    "fire_offset_s": fire_off,
                    "slot": slot_24,
                    "slot_canon": slot_canon,
                    "side": side,
                    "market_type": TYPE_FOR_SLOT[slot_canon],
                    "market_label": MARKET_LABELS[slot_canon],
                    "outcome": outcome_side,
                    "sc_ask": sc_ask, "sc_pred": float(sc_pred[slot_24]),
                    "sc_edge": sc_edge, "sc_fired": int(sc_fired),
                    "tx_ask": tx_ask, "tx_pred": float(tx_pred[slot_24]),
                    "tx_edge": tx_edge, "tx_fired": int(tx_fired),
                    "live_ask": live["ask"] if live else "",
                    "live_pred": live["predicted_prob"] if live else "",
                    "live_edge": live["edge"] if live else "",
                    "live_status": live["status"] if live else "",
                })

    # ---- report ----
    print(f"\n{'='*78}")
    print(f"Backtest replay PnL on full SC∩TX∩LIVE overlap "
          f"({len(sc_mids_by_slug)} games)")
    print(f"errors counted as valid trades (most are 403 geo-block from bot's VPN)")
    print(f"{'='*78}")
    print(f"{'source':<6s}  {'fires':>6s}  {'wins':>5s}  {'losses':>7s}  "
          f"{'win%':>6s}  {'PnL/share':>10s}  {'total PnL':>10s}  "
          f"{'unresolved':>10s}")
    for src in ("LIVE", "SC", "TX"):
        d = summary[src]
        resolved = d["wins"] + d["losses"]
        win_pct = d["wins"] / resolved * 100 if resolved else 0
        avg = d["pnl"] / resolved if resolved else 0
        print(f"{src:<6s}  {d['fires']:>6d}  {d['wins']:>5d}  {d['losses']:>7d}  "
              f"{win_pct:>5.1f}%  {avg:>+10.4f}  {d['pnl']:>+10.4f}  "
              f"{d['unresolved']:>10d}")
    live = summary["LIVE"]
    print(f"\nLIVE breakdown: {live['matched_fires']} matched + "
          f"{live['errored_fires']} errored "
          f"(+ {live['fires'] - live['matched_fires'] - live['errored_fires']} other)")

    # ---- write per-cell CSV ----
    if records:
        import csv as _csv
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        cols = list(records[0].keys())
        with OUT_CSV.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in records:
                # Format floats nicely
                for k, v in list(r.items()):
                    if isinstance(v, float):
                        r[k] = f"{v:.4f}"
                w.writerow(r)
        print(f"\nwrote per-cell CSV: {OUT_CSV} ({len(records)} rows)")


if __name__ == "__main__":
    main()
