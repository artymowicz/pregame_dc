"""Q3: backtest replay vs live fires.

For each (game, fire_offset_s) the live bot attempted, replay the model's
fire decision using SC data and TX data independently. Compares which
slots fire in each of {live, SC backtest, TX backtest}.

Per (game, slot, side) cell across the bot's enabled markets (moneyline,
totals), records:
  - live_fired, live_ask, live_pred, live_edge, live_status
  - sc_fired, sc_ask, sc_pred, sc_edge
  - tx_fired, tx_ask, tx_pred, tx_edge
where sc_fired = (ASK_LO <= sc_ask <= ASK_HI) & (sc_edge > THRESHOLD).
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pregame_dc import paths
from pregame_dc.constants import TYPE_FOR_SLOT, MARKET_LABELS
from pregame_dc.pipelines.helpers import canonical_slot_market_ids

LIVE_RESOLVED = paths.PACKAGE_ROOT / "pregame_dc" / "live" / "logs" / "orders_summary_resolved.jsonl"
SC_DIR = paths.PACKAGE_ROOT / "data" / "self_collected" / "per_game_data"
TX_DIR = paths.PACKAGE_ROOT / "data" / "telonex" / "per_game_data"
OUT_CSV = paths.PACKAGE_ROOT / "plots" / "backtest_replay_vs_live.csv"

THRESHOLD = 0.04
ASK_LO, ASK_HI = 0.01, 0.99
ENABLED_MARKETS = {"moneyline", "totals"}  # from live bot's attempt distribution
ENABLED_SLOTS = [s for s, mt in TYPE_FOR_SLOT.items() if mt in ENABLED_MARKETS]
ENABLED_24 = ENABLED_SLOTS + [s + 12 for s in ENABLED_SLOTS]   # YES + NO sides

MODEL_BY_OFFSET = {
    -1500: paths.MODEL_T_25MIN,
    -600:  paths.MODEL_T_10MIN,
}


def load_model(npz_path):
    d = np.load(npz_path)
    mu = d["mu"]
    sd = d["sd_safe"]
    beta_K = d["beta_K"]  # (24, 25)

    def predict(x_24):
        z = (x_24 - mu) / sd
        F = np.concatenate([z, [1.0]])
        return beta_K @ F   # (24,)
    return predict


def build_tx_index():
    """{bare_slug: path} by stripping -YYYY-MM-DD suffix from filename."""
    idx = {}
    for p in TX_DIR.glob("*.parquet"):
        m = re.match(r"^(.+)-\d{4}-\d{2}-\d{2}$", p.stem)
        if m:
            idx[m.group(1)] = p
        else:
            idx[p.stem] = p
    return idx


def ask_at_t(df, market_id, is_yes, t_ms):
    """Forward-fill best_ask for (market_id, is_yes) at t_ms.
    Returns ask, or 1.0 (placeholder) if no obs <= t_ms."""
    sub = df[(df["market_id"] == market_id) & (df["is_yes"] == is_yes)
             & (df["timestamp_ms"] <= t_ms)]
    if sub.empty:
        return 1.0
    row = sub.iloc[sub["timestamp_ms"].values.argmax()]
    return float(row["best_ask"])


def vector_24(df, slot_mids, t_ms):
    """Forward-fill 24-vector at t_ms. Placeholders default to 1.0.
    slots 0..11 = YES asks, slots 12..23 = NO asks."""
    x = np.full(24, 1.0)
    for slot_local, mid in enumerate(slot_mids):
        x[slot_local] = ask_at_t(df, mid, True, t_ms)
        x[slot_local + 12] = ask_at_t(df, mid, False, t_ms)
    return x


def derive_slot_mids(df):
    """From per-game parquet, build the canonical 12-slot market_id list.
    Returns None if game doesn't have all 12 canonical slots."""
    unique = df.groupby("market_id").first().reset_index()
    records = unique.to_dict("records")
    return canonical_slot_market_ids(records)


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    tx_index = build_tx_index()

    # Load live attempts
    attempts = []
    with LIVE_RESOLVED.open() as f:
        for line in f:
            if line.strip():
                attempts.append(json.loads(line))

    # Group attempts by (slug, fire_offset_s)
    by_game = defaultdict(list)
    for a in attempts:
        by_game[(a["game_slug"], a["fire_offset_s"])].append(a)

    # Load both models
    models = {off: load_model(p) for off, p in MODEL_BY_OFFSET.items()}

    rows = []
    cols = [
        "game_slug", "fire_offset_s", "fire_time_ms", "slot", "side", "market_type", "market_label",
        "in_sc", "in_tx",
        "live_fired", "live_ask", "live_pred", "live_edge", "live_status", "live_outcome", "live_pnl",
        "sc_fired", "sc_ask", "sc_pred", "sc_edge",
        "tx_fired", "tx_ask", "tx_pred", "tx_edge",
    ]
    stats = {"sc_only_games": 0, "tx_only_games": 0, "both_games": 0, "neither_games": 0,
             "sc_no_full_set": 0, "tx_no_full_set": 0}

    for (slug, fire_off), atts in by_game.items():
        fire_time_s = atts[0]["game_start_ts"] + fire_off
        fire_time_ms = fire_time_s * 1000
        live_by_slot = {(a["slot"], a["side"]): a for a in atts}

        # SC data
        sc_path = SC_DIR / f"{slug}.parquet"
        sc_df = pq.read_table(sc_path).to_pandas() if sc_path.exists() else None
        tx_path = tx_index.get(slug)
        tx_df = pq.read_table(tx_path).to_pandas() if tx_path else None

        if sc_df is not None and tx_df is not None: stats["both_games"] += 1
        elif sc_df is not None: stats["sc_only_games"] += 1
        elif tx_df is not None: stats["tx_only_games"] += 1
        else: stats["neither_games"] += 1

        sc_mids = derive_slot_mids(sc_df) if sc_df is not None else None
        tx_mids = derive_slot_mids(tx_df) if tx_df is not None else None
        if sc_df is not None and sc_mids is None: stats["sc_no_full_set"] += 1
        if tx_df is not None and tx_mids is None: stats["tx_no_full_set"] += 1

        sc_x = vector_24(sc_df, sc_mids, fire_time_ms) if sc_mids else None
        tx_x = vector_24(tx_df, tx_mids, fire_time_ms) if tx_mids else None

        predict = models[fire_off]
        sc_pred = predict(sc_x) if sc_x is not None else None
        tx_pred = predict(tx_x) if tx_x is not None else None

        # Iterate enabled slots × sides (moneyline + totals = 14 slots: 7 YES + 7 NO)
        for slot_24 in ENABLED_24:
            side = "Y" if slot_24 < 12 else "N"
            slot_canon = slot_24 if slot_24 < 12 else slot_24 - 12
            market_type = TYPE_FOR_SLOT[slot_canon]
            market_label = MARKET_LABELS[slot_canon]

            live_a = live_by_slot.get((slot_canon, side))

            def cell(x, pred):
                if x is None: return None, None, None, None
                ask = float(x[slot_24])
                p = float(pred[slot_24])
                edge = p - ask
                fired = (ASK_LO <= ask <= ASK_HI) and (edge > THRESHOLD)
                return fired, ask, p, edge

            sc_f, sc_a, sc_p, sc_e = cell(sc_x, sc_pred)
            tx_f, tx_a, tx_p, tx_e = cell(tx_x, tx_pred)

            # Skip cells with no signal in any source (saves space)
            if not live_a and not sc_f and not tx_f:
                continue

            row = {
                "game_slug": slug, "fire_offset_s": fire_off,
                "fire_time_ms": fire_time_ms,
                "slot": slot_canon, "side": side, "market_type": market_type,
                "market_label": market_label + (" YES" if side == "Y" else " NO"),
                "in_sc": int(sc_df is not None), "in_tx": int(tx_df is not None),
                "live_fired": int(live_a is not None and live_a["status"] == "matched"),
                "live_ask": live_a["ask"] if live_a else "",
                "live_pred": live_a["predicted_prob"] if live_a else "",
                "live_edge": live_a["edge"] if live_a else "",
                "live_status": live_a["status"] if live_a else "",
                "live_outcome": (live_a.get("outcome") if live_a else "") or "",
                "live_pnl": (live_a.get("realised_pnl") if live_a else "") or "",
                "sc_fired": "" if sc_f is None else int(sc_f),
                "sc_ask": "" if sc_a is None else f"{sc_a:.4f}",
                "sc_pred": "" if sc_p is None else f"{sc_p:.4f}",
                "sc_edge": "" if sc_e is None else f"{sc_e:+.4f}",
                "tx_fired": "" if tx_f is None else int(tx_f),
                "tx_ask": "" if tx_a is None else f"{tx_a:.4f}",
                "tx_pred": "" if tx_p is None else f"{tx_p:.4f}",
                "tx_edge": "" if tx_e is None else f"{tx_e:+.4f}",
            }
            rows.append([row[c] for c in cols])

    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)

    # Summary
    print(f"unique (game, fire_offset_s) pairs: {len(by_game)}")
    print(f"per-game data coverage: {stats}")
    print(f"wrote {OUT_CSV} ({len(rows):,} cells with at least one signal)")

    # Agreement stats — restrict to cells where both SC and TX have data
    ci = lambda name: cols.index(name)
    common = [r for r in rows if r[ci("in_sc")] == 1 and r[ci("in_tx")] == 1
              and r[ci("sc_fired")] != "" and r[ci("tx_fired")] != ""]
    live_attempted = [r for r in common if r[ci("live_status")] != ""]
    live_matched   = [r for r in common if r[ci("live_fired")] == 1]
    sc_fired       = [r for r in common if r[ci("sc_fired")] == 1]
    tx_fired       = [r for r in common if r[ci("tx_fired")] == 1]
    print(f"\nOn cells where both SC and TX have data:")
    print(f"  total cells with any signal: {len(common):,}")
    print(f"  live attempted:              {len(live_attempted):,}")
    print(f"  live matched:                {len(live_matched):,}")
    print(f"  SC backtest would fire:      {len(sc_fired):,}")
    print(f"  TX backtest would fire:      {len(tx_fired):,}")

    # 3-way set comparison
    live_keys = {(r[ci("game_slug")], r[ci("slot")], r[ci("side")]) for r in common
                 if r[ci("live_status")] != ""}
    live_matched_keys = {(r[ci("game_slug")], r[ci("slot")], r[ci("side")]) for r in common
                         if r[ci("live_fired")] == 1}
    sc_keys = {(r[ci("game_slug")], r[ci("slot")], r[ci("side")]) for r in common
               if r[ci("sc_fired")] == 1}
    tx_keys = {(r[ci("game_slug")], r[ci("slot")], r[ci("side")]) for r in common
               if r[ci("tx_fired")] == 1}

    print(f"\n3-way fire-set comparison on SC∩TX games:")
    print(f"  live attempted: {len(live_keys)}, live matched: {len(live_matched_keys)}, "
          f"SC: {len(sc_keys)}, TX: {len(tx_keys)}")
    print(f"  live attempted ∩ SC fired: {len(live_keys & sc_keys)}")
    print(f"  live attempted ∩ TX fired: {len(live_keys & tx_keys)}")
    print(f"  SC fired ∩ TX fired:       {len(sc_keys & tx_keys)}")
    print(f"  live ∩ SC ∩ TX (all 3):    {len(live_keys & sc_keys & tx_keys)}")
    print(f"  live but neither SC nor TX:{len(live_keys - sc_keys - tx_keys)}")
    print(f"  SC but not live or TX:     {len(sc_keys - live_keys - tx_keys)}")
    print(f"  TX but not live or SC:     {len(tx_keys - live_keys - sc_keys)}")


if __name__ == "__main__":
    main()
