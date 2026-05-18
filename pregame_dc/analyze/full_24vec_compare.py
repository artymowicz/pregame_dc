"""Per-game CSV: full 24-vector at fire time from each of SC, TX, LIVE.

Scope: {games where live test matched a fire} ∩ {in self_collected} ∩ {in telonex}.

For each such game, emit three rows (SC, TX, LIVE) with the 24-element ask
vector and the 12-element model prediction (YES side; NO pred = 1 - YES).

Live values are parsed from bot_stdout.log "ask(YES)" / "ask(NO)" / "pred"
lines that the bot emits at fire time.
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
from pregame_dc.constants import X_COLS, MARKET_LABELS
from pregame_dc.pipelines.helpers import canonical_slot_market_ids

LIVE_RESOLVED = paths.PACKAGE_ROOT / "pregame_dc" / "live" / "logs" / "orders_summary_resolved.jsonl"
BOT_STDOUT = paths.PACKAGE_ROOT / "pregame_dc" / "live" / "logs" / "bot_stdout.log"
SC_DIR = paths.PACKAGE_ROOT / "data" / "self_collected" / "per_game_data"
TX_DIR = paths.PACKAGE_ROOT / "data" / "telonex" / "per_game_data"
OUT_CSV = paths.PACKAGE_ROOT / "plots" / "full_24vec_compare.csv"

MODEL_BY_OFFSET = {
    -1500: paths.MODEL_T_25MIN,
    -600:  paths.MODEL_T_10MIN,
}


def load_model(npz_path):
    d = np.load(npz_path)
    mu, sd, beta_K = d["mu"], d["sd_safe"], d["beta_K"]

    def predict(x_24):
        z = (x_24 - mu) / sd
        F = np.concatenate([z, [1.0]])
        return beta_K @ F   # (24,)
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


def vector_24_from_pg(df, slot_mids, t_ms):
    x = np.full(24, 1.0)
    for slot_local, mid in enumerate(slot_mids):
        x[slot_local] = ask_at_t(df, mid, True, t_ms)
        x[slot_local + 12] = ask_at_t(df, mid, False, t_ms)
    return x


def derive_slot_mids(df):
    unique = df.groupby("market_id").first().reset_index()
    records = unique.to_dict("records")
    return canonical_slot_market_ids(records)


def _detect_bot_tz_offset_ms(stdout_path: Path, attempts):
    """The bot's `now_str()` uses naive `datetime.now()` → local time on the
    bot's machine. Detect that offset from UTC by matching each matched fire's
    `ts_placed` (UTC ISO in orders_summary) with its corresponding
    `FIRE {slug} ...` stdout line. Returns offset in ms such that
        utc_ms = local_parsed_ms - offset_ms.
    """
    from datetime import datetime, timezone

    ts_local_re = re.compile(
        r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]\s+FIRE\s+(\S+)")
    fire_local = {}   # slug -> first local stdout fire ts (ms)
    with stdout_path.open() as f:
        for line in f:
            m = ts_local_re.match(line)
            if m and m.group(2) not in fire_local:
                fire_local[m.group(2)] = int(
                    datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
                    .timestamp() * 1000)

    offsets = []
    for a in attempts:
        if a["status"] != "matched":
            continue
        slug = a["game_slug"]
        if slug not in fire_local:
            continue
        utc_ms = int(datetime.fromisoformat(a["ts_responded"]).timestamp() * 1000)
        offsets.append(fire_local[slug] - utc_ms)
    if not offsets:
        return 0
    return int(np.median(offsets))


def parse_live_24_vectors(path: Path, tz_offset_ms: int = 0):
    """Scan bot_stdout.log for the per-fire-event 24-vector + pred dump.

    Format (4 consecutive lines, all starting with '[ts]   {slug}'):
        [ts]   {slug} markets:    A_win B_win Draw  A-1.5 ...
        [ts]   {slug}    ask(YES): v0 v1 ... v11
        [ts]   {slug}    ask(NO):  v12 v13 ... v23
        [ts]   {slug}    pred:     p0 p1 ... p11

    The bot reads MarketTracker state at the moment it prints the
    ask(YES) line (after _wait_for_books succeeds); we capture that
    timestamp so SC/TX can be forward-filled to the same instant rather
    than the nominal fire_time = game_start + fire_offset_s.

    Stdout timestamps are naive local time (bot's `now_str` calls
    `datetime.now()`), so `tz_offset_ms` is subtracted to get UTC ms.

    Returns {slug: {"x": (24,), "pred12": (12,), "read_ts_ms": int}}.
    Last occurrence wins if the bot re-fired the same slug.
    """
    out = {}
    ts_prefix = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]")
    asks_y_re = re.compile(r"\]   (\S+)    ask\(YES\):\s+(.+)$")
    asks_n_re = re.compile(r"\]   (\S+)    ask\(NO\):\s+(.+)$")
    pred_re   = re.compile(r"\]   (\S+)    pred:\s+(.+)$")

    from datetime import datetime

    def parse_ts_ms(line):
        m = ts_prefix.match(line)
        if not m:
            return None
        local_ms = int(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
                       .timestamp() * 1000)
        return local_ms - tz_offset_ms

    state = {}
    with path.open() as f:
        for line in f:
            m = asks_y_re.search(line)
            if m:
                slug = m.group(1)
                vals = [float(v) for v in m.group(2).split()]
                if len(vals) == 12:
                    state.setdefault(slug, {})["yes"] = vals
                    state[slug]["read_ts_ms"] = parse_ts_ms(line)
                continue
            m = asks_n_re.search(line)
            if m:
                slug = m.group(1)
                vals = [float(v) for v in m.group(2).split()]
                if len(vals) == 12:
                    state.setdefault(slug, {})["no"] = vals
                continue
            m = pred_re.search(line)
            if m:
                slug = m.group(1)
                vals = [float(v) for v in m.group(2).split()]
                if len(vals) == 12:
                    state.setdefault(slug, {})["pred"] = vals
                    d = state[slug]
                    if "yes" in d and "no" in d and "pred" in d:
                        out[slug] = {
                            "x": np.array(d["yes"] + d["no"], dtype=np.float64),
                            "pred12": np.array(d["pred"], dtype=np.float64),
                            "read_ts_ms": d.get("read_ts_ms"),
                        }
                        state[slug] = {}
    return out


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    tx_index = build_tx_index()
    sc_slugs = {p.stem for p in SC_DIR.glob("*.parquet")}

    # Live matched fires
    matched = []
    with LIVE_RESOLVED.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["status"] == "matched":
                    matched.append(r)

    fired_slugs = sorted({r["game_slug"] for r in matched})
    fired_overlap = sorted(set(fired_slugs) & sc_slugs & set(tx_index))
    print(f"live matched-fire games: {len(fired_slugs)}")
    print(f"  ∩ SC ({len(sc_slugs)} files):    {len(set(fired_slugs) & sc_slugs)}")
    print(f"  ∩ TX ({len(tx_index)} files):    {len(set(fired_slugs) & set(tx_index))}")
    print(f"  3-way intersection:    {len(fired_overlap)}")

    fire_offset_by_slug = {r["game_slug"]: r["fire_offset_s"] for r in matched}
    fire_start_ts_by_slug = {r["game_slug"]: r["game_start_ts"] for r in matched}

    tz_offset_ms = _detect_bot_tz_offset_ms(BOT_STDOUT, matched)
    print(f"\ndetected bot stdout tz offset from UTC: "
          f"{tz_offset_ms/3600/1000:+.2f} h (local - UTC)")

    print("parsing live 24-vectors from bot_stdout.log ...")
    live_vecs = parse_live_24_vectors(BOT_STDOUT, tz_offset_ms=tz_offset_ms)
    print(f"  found {len(live_vecs)} games with full 24-vector + pred in stdout")
    n_overlap_with_live = sum(1 for s in fired_overlap if s in live_vecs)
    print(f"  of overlap games: {n_overlap_with_live} / {len(fired_overlap)} have a stdout 24-vector")

    models = {off: load_model(p) for off, p in MODEL_BY_OFFSET.items()}

    cols = ["game_slug", "fire_offset_s", "live_read_ts_ms", "source"] \
        + X_COLS \
        + [f"pred_{i}" for i in range(12)]   # YES-side preds only (NO = 1 - YES by construction)

    rows = []
    for slug in fired_overlap:
        fire_off = fire_offset_by_slug[slug]
        nominal_fire_ms = (fire_start_ts_by_slug[slug] + fire_off) * 1000
        live = live_vecs.get(slug)
        # Use the bot's actual read timestamp (when it printed ask(YES)) so
        # SC/TX are snapshotted at the same instant as the MarketTracker read.
        # Fallback to nominal fire_time if not parsed.
        read_ts_ms = (live["read_ts_ms"] if live and live.get("read_ts_ms") is not None
                      else nominal_fire_ms)
        predict = models[fire_off]

        # SC
        sc_df = pq.read_table(SC_DIR / f"{slug}.parquet").to_pandas()
        sc_mids = derive_slot_mids(sc_df)
        if sc_mids is None:
            print(f"  WARN: {slug} has no canonical 12-slot set in SC; skipping SC")
            sc_x = None
        else:
            sc_x = vector_24_from_pg(sc_df, sc_mids, read_ts_ms)
        sc_pred = predict(sc_x)[:12] if sc_x is not None else None

        # TX
        tx_df = pq.read_table(tx_index[slug]).to_pandas()
        tx_mids = derive_slot_mids(tx_df)
        if tx_mids is None:
            print(f"  WARN: {slug} has no canonical 12-slot set in TX; skipping TX")
            tx_x = None
        else:
            tx_x = vector_24_from_pg(tx_df, tx_mids, read_ts_ms)
        tx_pred = predict(tx_x)[:12] if tx_x is not None else None

        # LIVE
        live = live_vecs.get(slug)
        live_x = live["x"] if live else None
        live_pred = live["pred12"] if live else None

        def row_for(source, x, pred12):
            r = {"game_slug": slug, "fire_offset_s": fire_off,
                 "live_read_ts_ms": read_ts_ms, "source": source}
            for i, col in enumerate(X_COLS):
                r[col] = "" if x is None else f"{x[i]:.4f}"
            for i in range(12):
                r[f"pred_{i}"] = "" if pred12 is None else f"{pred12[i]:.4f}"
            return [r[c] for c in cols]

        rows.append(row_for("SC", sc_x, sc_pred))
        rows.append(row_for("TX", tx_x, tx_pred))
        rows.append(row_for("LIVE", live_x, live_pred))

    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    print(f"\nwrote {OUT_CSV}")
    print(f"games: {len(fired_overlap)}, rows: {len(rows)} (3 per game: SC, TX, LIVE)")

    # Quick summary: agreement on the 24-vectors among the overlap
    print("\n=== Per-vector agreement on overlap games (where all three sources have a vector) ===")
    triples = []
    x_off = cols.index("x_0")
    for i in range(0, len(rows), 3):
        sc_row, tx_row, live_row = rows[i], rows[i+1], rows[i+2]
        if all(sc_row[x_off+j] != "" for j in range(24)) \
           and all(tx_row[x_off+j] != "" for j in range(24)) \
           and all(live_row[x_off+j] != "" for j in range(24)):
            sc_v = np.array([float(sc_row[x_off+j]) for j in range(24)])
            tx_v = np.array([float(tx_row[x_off+j]) for j in range(24)])
            lv_v = np.array([float(live_row[x_off+j]) for j in range(24)])
            triples.append((sc_row[0], sc_v, tx_v, lv_v))
    print(f"games with all-three full 24-vectors: {len(triples)}")
    if triples:
        sc_stack = np.array([t[1] for t in triples])
        tx_stack = np.array([t[2] for t in triples])
        lv_stack = np.array([t[3] for t in triples])
        for name, a, b in [("live - TX", lv_stack, tx_stack),
                           ("live - SC", lv_stack, sc_stack),
                           ("SC - TX",   sc_stack, tx_stack)]:
            d = a - b
            ad = np.abs(d)
            # Ignore placeholder cells (where ask == 1.0 in either source)
            valid = (a < 1.0) & (b < 1.0)
            d_v = d[valid]
            ad_v = np.abs(d_v)
            print(f"  {name}: cells={d.size} valid(both<1.0)={valid.sum()} "
                  f"mean_diff={d_v.mean():+.4f} p50|d|={np.median(ad_v):.4f} "
                  f"p95|d|={np.percentile(ad_v, 95):.4f} "
                  f"exact={(d_v == 0).mean()*100:.1f}% (of valid)")


if __name__ == "__main__":
    main()
