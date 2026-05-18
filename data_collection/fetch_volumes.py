"""Fetch traded-volume stats per market from Gamma into data/volumes.parquet.

Volume is NOT returned by the per-market endpoint `/markets/{id}`. It IS
returned by the list endpoint `/markets?closed=true&id=X`, which is what
this script uses.

Source of market_ids: union of data/self_collected/games.csv and
data/telonex/games.csv. Resumable: rows already present (with non-null
volume) are kept on subsequent runs.

Output schema:
    market_id     int64
    volume        float64    (volumeNum = lifetime traded $)
    volume_clob   float64    (CLOB-only lifetime)
    volume_amm    float64    (AMM-only lifetime; usually 0 for new markets)
    volume_24hr   float64
    volume_1wk    float64
    volume_1mo    float64
    volume_1yr    float64
    fetched       bool       (true even if vol fields were null, to skip refetch)

Usage:
    python -m data_collection.fetch_volumes
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from pregame_dc import paths

GAMMA_URL = "https://gamma-api.polymarket.com/markets"


def fetch_one(session: requests.Session, market_id: int) -> dict:
    for attempt in range(4):
        try:
            r = session.get(
                GAMMA_URL,
                params={"closed": "true", "id": str(market_id), "limit": 1},
                timeout=30,
            )
            if r.status_code == 404:
                return {"market_id": market_id, "fetched": True}
            r.raise_for_status()
            arr = r.json()
            if not arr:
                # Not closed yet, or filter didn't match
                return {"market_id": market_id, "fetched": True}
            m = arr[0]
            def f(k):
                v = m.get(k)
                return float(v) if v is not None else None
            return {
                "market_id": market_id,
                "volume": f("volumeNum"),
                "volume_clob": f("volumeClob"),
                "volume_amm": f("volumeAmm"),
                "volume_24hr": f("volume24hr"),
                "volume_1wk": f("volume1wk"),
                "volume_1mo": f("volume1mo"),
                "volume_1yr": f("volume1yr"),
                "fetched": True,
            }
        except (requests.RequestException, ValueError) as e:
            if attempt == 3:
                return {"market_id": market_id, "fetched": False,
                        "error": f"{type(e).__name__}"}
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
    ap.add_argument("--out", type=Path,
                    default=paths.DATA_DIR / "volumes.parquet")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    print("collecting market_ids ...")
    ids = sorted(_collect_market_ids())
    print(f"union market_ids: {len(ids)}")

    existing: dict[int, dict] = {}
    if args.out.exists() and not args.no_resume:
        df = pd.read_parquet(args.out)
        existing = {int(r.market_id): r._asdict()
                    for r in df.itertuples(index=False)}
        print(f"resumable: {len(existing)} previously-fetched")

    todo = [mid for mid in ids
            if mid not in existing or not existing[mid].get("fetched")]
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

    cols = ["market_id", "volume", "volume_clob", "volume_amm",
            "volume_24hr", "volume_1wk", "volume_1mo", "volume_1yr", "fetched"]
    rows = []
    for r in results.values():
        rows.append({c: r.get(c) for c in cols})
    out_df = pd.DataFrame(rows)
    out_df["market_id"] = out_df["market_id"].astype("int64")
    for c in cols[1:-1]:
        out_df[c] = out_df[c].astype("float64")
    out_df["fetched"] = out_df["fetched"].astype("bool")

    n_with_vol = out_df["volume"].notna().sum()
    print(f"\nfinal: {n_with_vol}/{len(out_df)} have volume "
          f"({n_with_vol/len(out_df):.1%})")
    print(f"volume describe:\n{out_df['volume'].describe()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
