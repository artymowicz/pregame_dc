"""Per-fire comparison: live bot's recorded ask vs SC backtest vs TX backtest.

For each matched fire in logs/orders_summary_resolved.jsonl, find the same
(game_slug, market_id, side) in the self_collected and telonex per-game
parquets and forward-fill to the fire timestamp to get the
backtest-equivalent ask and bid. Output one row per fire to CSV.

Goal: distinguish "data quality" vs "distributional difference" as the
source of the live-vs-backtest PnL gap. If live recorded asks match SC
and TX asks at fire time, the gap is distributional. Where they diverge,
data quality is implicated.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pregame_dc import paths

LIVE_RESOLVED = paths.PACKAGE_ROOT / "pregame_dc" / "live" / "logs" / "orders_summary_resolved.jsonl"
SC_DIR = paths.PACKAGE_ROOT / "data" / "self_collected" / "per_game_data"
TX_DIR = paths.PACKAGE_ROOT / "data" / "telonex" / "per_game_data"
OUT_CSV = paths.PACKAGE_ROOT / "plots" / "live_vs_backtest_asks.csv"


def _at_t(df, market_id, is_yes, t_ms):
    """Forward-fill: row with max timestamp_ms <= t_ms for (market_id, is_yes).
    Returns (best_ask, best_bid, best_ask_size, best_bid_size, n_obs_before, last_ts_ms)
    or all None if no row exists."""
    sub = df[(df["market_id"] == market_id) & (df["is_yes"] == is_yes)
             & (df["timestamp_ms"] <= t_ms)]
    if sub.empty:
        return None, None, None, None, 0, None
    row = sub.iloc[sub["timestamp_ms"].values.argmax()]
    return (float(row["best_ask"]), float(row["best_bid"]),
            float(row["best_ask_size"]), float(row["best_bid_size"]),
            len(sub), int(row["timestamp_ms"]))


def main():
    # Live fires
    fires = []
    with LIVE_RESOLVED.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["status"] == "matched":
                    fires.append(r)
    print(f"matched live fires: {len(fires)}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "game_slug", "market_id", "slot", "side", "market_type", "market_label",
        "fire_offset_s", "fire_time_ms",
        "live_ask", "live_ask_complement", "live_pred_p", "live_edge",
        "live_filled_size", "live_outcome", "live_pnl",
        "sc_present", "sc_ask", "sc_bid", "sc_ask_size", "sc_bid_size",
        "sc_n_obs_before", "sc_last_obs_lag_s",
        "tx_present", "tx_ask", "tx_bid", "tx_ask_size", "tx_bid_size",
        "tx_n_obs_before", "tx_last_obs_lag_s",
    ]

    # Cache per-game data
    sc_cache = {}
    tx_cache = {}

    def get_pg(slug, src_dir, cache):
        if slug in cache:
            return cache[slug]
        p = src_dir / f"{slug}.parquet"
        if not p.exists():
            cache[slug] = None
            return None
        df = pq.read_table(p).to_pandas()
        cache[slug] = df
        return df

    rows = []
    n_sc = n_tx = n_both = 0
    for fire in fires:
        slug = fire["game_slug"]
        market_id = fire["market_id"]
        is_yes = fire["side"] == "Y"
        fire_time_s = fire["game_start_ts"] + fire["fire_offset_s"]
        fire_time_ms = fire_time_s * 1000

        sc_df = get_pg(slug, SC_DIR, sc_cache)
        tx_df = get_pg(slug, TX_DIR, tx_cache)
        sc_present = sc_df is not None
        tx_present = tx_df is not None
        if sc_present: n_sc += 1
        if tx_present: n_tx += 1
        if sc_present and tx_present: n_both += 1

        row = {
            "game_slug": slug, "market_id": market_id, "slot": fire["slot"],
            "side": fire["side"], "market_type": fire["market_type"],
            "market_label": fire["market_label"],
            "fire_offset_s": fire["fire_offset_s"], "fire_time_ms": fire_time_ms,
            "live_ask": fire["ask"], "live_ask_complement": fire["ask_complement"],
            "live_pred_p": fire["predicted_prob"], "live_edge": fire["edge"],
            "live_filled_size": fire["filled_size"],
            "live_outcome": fire["outcome"], "live_pnl": fire["realised_pnl"],
            "sc_present": int(sc_present), "tx_present": int(tx_present),
        }
        for src, df, prefix in [(sc_df, "sc"), (tx_df, "tx")]:
            if df is None:
                row[f"{prefix}_ask"] = ""
                row[f"{prefix}_bid"] = ""
                row[f"{prefix}_ask_size"] = ""
                row[f"{prefix}_bid_size"] = ""
                row[f"{prefix}_n_obs_before"] = ""
                row[f"{prefix}_last_obs_lag_s"] = ""
                continue
            ask, bid, asz, bsz, n_obs, last_ts = _at_t(df, market_id, is_yes, fire_time_ms)
            row[f"{prefix}_ask"] = "" if ask is None else f"{ask:.4f}"
            row[f"{prefix}_bid"] = "" if bid is None else f"{bid:.4f}"
            row[f"{prefix}_ask_size"] = "" if asz is None else f"{asz:.1f}"
            row[f"{prefix}_bid_size"] = "" if bsz is None else f"{bsz:.1f}"
            row[f"{prefix}_n_obs_before"] = n_obs
            row[f"{prefix}_last_obs_lag_s"] = "" if last_ts is None else f"{(fire_time_ms-last_ts)/1000:.1f}"

        # Re-arrange to match cols order
        rows.append([row.get(c, "") for c in cols])

    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)

    print(f"\nwrote {OUT_CSV}")
    print(f"per-game data coverage: SC={n_sc}/{len(fires)}, TX={n_tx}/{len(fires)}, BOTH={n_both}/{len(fires)}")

    # Summary stats
    print("\n=== Summary on cells where both SC and TX have data ===")
    valid_rows = [r for r in rows if r[cols.index("sc_present")] == 1 and r[cols.index("tx_present")] == 1
                  and r[cols.index("sc_ask")] != "" and r[cols.index("tx_ask")] != ""]
    print(f"matched live fires with SC+TX backtest data: {len(valid_rows)}")
    if valid_rows:
        def vec(name):
            return np.array([float(r[cols.index(name)]) for r in valid_rows])
        live = vec("live_ask")
        sc = vec("sc_ask")
        tx = vec("tx_ask")
        print(f"\n{'pair':<20s} {'mean diff':>11s} {'p50 |d|':>9s} {'p95 |d|':>9s} {'p99 |d|':>9s} {'%agree (|d|<0.005)':>20s}")
        for name, a, b in [
            ("live - SC", live, sc), ("live - TX", live, tx), ("SC - TX", sc, tx),
        ]:
            d = a - b
            ad = np.abs(d)
            print(f"{name:<20s} {d.mean():>+11.4f} {np.percentile(ad,50):>9.4f} "
                  f"{np.percentile(ad,95):>9.4f} {np.percentile(ad,99):>9.4f} "
                  f"{(ad<0.005).mean()*100:>19.1f}%")


if __name__ == "__main__":
    main()
