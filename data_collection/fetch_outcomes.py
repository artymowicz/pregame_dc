"""Fetch resolved-market outcomes from Gamma into data/outcomes.parquet.

The labeled-dataset builder reads `data/outcomes.parquet` to map each canonical
slot's market_id to its resolved YES-side price (0/1). The shipped file covers
games seen at export time; new games collected via data_collection/ need their
outcomes appended here before they show up in
`pregame_dc.pipelines.build_labeled_dataset`.

Source of market_ids is the union of:
    data/self_collected/games.csv   (written by data_collection.self_collected.build_dataset)
    data/telonex/games.csv          (written by data_collection.telonex.build_dataset)

Output schema (matches data/outcomes.parquet shipped with the repo):
    market_id    int64
    final_price  float64    (0.0/1.0 if closed and resolved; NaN otherwise)
    closed       bool
    closed_time  string     (ISO8601)
    uma_status   string

Resumable: rows already present in the output parquet are kept (unless
--no-resume). Concurrent: ThreadPoolExecutor, default 20 workers.

Usage:
    python -m data_collection.fetch_outcomes
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from pregame_dc import paths

GAMMA_URL = "https://gamma-api.polymarket.com/markets/{id}"
CLOB_URL = "https://clob.polymarket.com/markets/{cid}"


def _fetch_clob(session: requests.Session, market_id: int,
                condition_id: str) -> dict | None:
    """CLOB fallback for markets that 404 on Gamma (e.g. silently renumbered
    market_ids). Returns the same output schema as fetch_one, or None if CLOB
    also misses (pre-2023 markets are not indexed in CLOB). See
    docs/polymarket_market_metadata.md section 2b."""
    url = CLOB_URL.format(cid=condition_id)
    for attempt in range(4):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            m = r.json()
            closed = bool(m.get("closed", False))
            # Per docs/polymarket_market_metadata.md: tokens[0] is the YES side
            # by Polymarket convention (clobTokenIds[0]). The `outcome` field
            # is the human-readable label (literal "Yes" for binary markets,
            # but team names for 3-way moneylines and team-named spreads), so
            # filtering on outcome=="Yes" silently misses team-named markets.
            tokens = m.get("tokens") or []
            yes_price = None
            if tokens and tokens[0].get("price") is not None:
                try:
                    yes_price = float(tokens[0]["price"])
                except (TypeError, ValueError):
                    pass
            final_price = yes_price if (closed and yes_price is not None) else None
            # CLOB has no closedTime / umaResolutionStatus equivalents — we
            # mark uma_status to record that the value came from CLOB so
            # downstream consumers can tell.
            return {"market_id": market_id, "final_price": final_price,
                    "closed": closed, "closed_time": "",
                    "uma_status": "resolved_clob" if final_price is not None else "clob_open"}
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt == 3:
                return {"market_id": market_id, "final_price": None,
                        "closed": False, "closed_time": "",
                        "uma_status": f"error_clob:{type(e).__name__}"}
            time.sleep(1.5 ** attempt)


def fetch_one(session: requests.Session, market_id: int,
              condition_id: str | None = None) -> dict:
    """Fetch one market via Gamma; fall back to CLOB on 404 when condition_id
    is provided. Polymarket silently renumbers market_id over time; the
    condition_id is stable, so CLOB can recover renumbered markets."""
    url = GAMMA_URL.format(id=market_id)
    for attempt in range(4):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 404:
                if condition_id:
                    clob = _fetch_clob(session, market_id, condition_id)
                    if clob is not None:
                        return clob
                return {"market_id": market_id, "final_price": None,
                        "closed": False, "closed_time": "", "uma_status": "not_found"}
            r.raise_for_status()
            m = r.json()
            try:
                prices = json.loads(m.get("outcomePrices", "[]"))
                yes_price = float(prices[0]) if prices else None
            except (json.JSONDecodeError, ValueError, IndexError):
                yes_price = None
            closed = bool(m.get("closed", False))
            closed_time = m.get("closedTime", "") or ""
            uma_status = m.get("umaResolutionStatus", "") or ""
            final_price = yes_price if (closed and yes_price is not None) else None
            return {"market_id": market_id, "final_price": final_price,
                    "closed": closed, "closed_time": closed_time,
                    "uma_status": uma_status}
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt == 3:
                return {"market_id": market_id, "final_price": None,
                        "closed": False, "closed_time": "",
                        "uma_status": f"error:{type(e).__name__}"}
            time.sleep(1.5 ** attempt)


def _collect_market_ids() -> dict[int, str]:
    """Return {market_id: condition_id} across all games.csv files. When the
    same market_id appears with conflicting condition_ids (shouldn't happen,
    but defensively), last-write-wins."""
    out: dict[int, str] = {}
    for csv_path in (paths.SELF_COLLECTED_GAMES_CSV, paths.TELONEX_GAMES_CSV):
        if not csv_path.exists():
            print(f"  (skipping {csv_path}: not present)")
            continue
        df = pd.read_csv(csv_path, usecols=["market_id", "condition_id"]).dropna()
        df["market_id"] = df["market_id"].astype(int)
        for mid, cid in zip(df["market_id"], df["condition_id"]):
            out[int(mid)] = str(cid)
        print(f"  {csv_path}: {df['market_id'].nunique()} unique market_ids")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=paths.OUTCOMES,
                    help=f"output parquet (default: {paths.OUTCOMES})")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: fetch only first N market_ids")
    ap.add_argument("--no-resume", action="store_true",
                    help="refetch all market_ids even if already in the output parquet")
    args = ap.parse_args()

    print("collecting market_ids ...")
    mid_to_cid = _collect_market_ids()
    if not mid_to_cid:
        raise SystemExit(
            "no market_ids found. Run data_collection.self_collected.build_dataset "
            "or data_collection.telonex.build_dataset first to populate the "
            "games.csv files."
        )
    ids = sorted(mid_to_cid.keys())
    print(f"union market_ids: {len(ids)}")

    existing: dict[int, dict] = {}
    if args.out.exists() and not args.no_resume:
        df = pd.read_parquet(args.out)
        existing = {int(r.market_id): r._asdict()
                    for r in df.itertuples(index=False)}
        already_resolved = sum(
            1 for mid in ids
            if mid in existing and pd.notna(existing[mid].get("final_price"))
        )
        print(f"resumable: {len(existing)} previously-fetched, "
              f"{already_resolved} resolved")

    todo = [mid for mid in ids
            if mid not in existing or pd.isna(existing[mid].get("final_price"))]
    if args.limit:
        todo = todo[:args.limit]
    print(f"to fetch: {len(todo)}")

    if not todo:
        print("nothing to do")
        return

    results = dict(existing)
    session = requests.Session()
    session.headers["User-Agent"] = "pregame_dc-data_collection/1.0"

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(fetch_one, session, mid, mid_to_cid.get(mid)): mid
            for mid in todo
        }
        n_done = 0
        for fut in as_completed(futures):
            row = fut.result()
            results[row["market_id"]] = row
            n_done += 1
            if n_done % 200 == 0 or n_done == len(todo):
                rate = n_done / (time.time() - t0 + 1e-6)
                eta = (len(todo) - n_done) / rate if rate > 0 else 0
                print(f"  {n_done}/{len(todo)}  rate={rate:.1f}/s  eta={eta:.0f}s")

    out_df = pd.DataFrame(list(results.values()))
    out_df["market_id"] = out_df["market_id"].astype("int64")
    out_df["final_price"] = out_df["final_price"].astype("float64")
    out_df["closed"] = out_df["closed"].astype("bool")
    out_df["closed_time"] = out_df["closed_time"].astype("string")
    out_df["uma_status"] = out_df["uma_status"].astype("string")

    n_resolved = out_df["final_price"].notna().sum()
    print(f"\nfinal: {n_resolved}/{len(out_df)} resolved "
          f"({n_resolved/len(out_df):.1%})")
    print(f"uma_status counts:\n{out_df['uma_status'].value_counts()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
