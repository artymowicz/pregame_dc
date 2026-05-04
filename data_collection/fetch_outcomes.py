"""Fetch resolved-market outcomes from Gamma into data/outcomes.parquet.

The labeled-dataset builder reads `data/outcomes.parquet` to map each canonical
slot's market_id to its resolved YES-side price (0/1). The shipped file covers
games seen at export time; new games collected via data_collection/ need their
outcomes appended here before they show up in
`pregame_pca.pipelines.build_labeled_dataset`.

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

from pregame_pca import paths

GAMMA_URL = "https://gamma-api.polymarket.com/markets/{id}"


def fetch_one(session: requests.Session, market_id: int) -> dict:
    url = GAMMA_URL.format(id=market_id)
    for attempt in range(4):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 404:
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


def _collect_market_ids() -> set[int]:
    ids: set[int] = set()
    for csv_path in (paths.SELF_COLLECTED_GAMES_CSV, paths.TELONEX_GAMES_CSV):
        if not csv_path.exists():
            print(f"  (skipping {csv_path}: not present)")
            continue
        ids_one = (
            pd.read_csv(csv_path, usecols=["market_id"])["market_id"]
              .dropna().astype(int).unique()
        )
        ids.update(ids_one.tolist())
        print(f"  {csv_path}: {len(ids_one)} unique market_ids")
    return ids


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
    ids = sorted(_collect_market_ids())
    if not ids:
        raise SystemExit(
            "no market_ids found. Run data_collection.self_collected.build_dataset "
            "or data_collection.telonex.build_dataset first to populate the "
            "games.csv files."
        )
    print(f"union market_ids: {len(ids)}")

    existing: dict[int, dict] = {}
    if args.out.exists() and not args.no_resume:
        df = pd.read_parquet(args.out)
        existing = {int(r.market_id): r._asdict()
                    for r in df.itertuples(index=False)}
        already_resolved = sum(
            1 for mid in ids
            if mid in existing and existing[mid].get("final_price") is not None
        )
        print(f"resumable: {len(existing)} previously-fetched, "
              f"{already_resolved} resolved")

    todo = [mid for mid in ids
            if mid not in existing or existing[mid].get("final_price") is None]
    if args.limit:
        todo = todo[:args.limit]
    print(f"to fetch: {len(todo)}")

    if not todo:
        print("nothing to do")
        return

    results = dict(existing)
    session = requests.Session()
    session.headers["User-Agent"] = "pregame_pca-data_collection/1.0"

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_one, session, mid): mid for mid in todo}
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
