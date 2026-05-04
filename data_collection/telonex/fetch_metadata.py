"""Download Telonex's Polymarket market roster (markets.parquet).

The roster is a ~540 MB parquet of every Polymarket market Telonex tracks
(market_id, slug, asset_id, type, condition_id, etc.). It is the input that
`data_collection.telonex.build_dataset` reads to discover which markets to
download quotes for. The endpoint is anonymous (no API key required).

Usage:
    python -m data_collection.telonex.fetch_metadata
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

from pregame_pca import paths

URL = "https://api.telonex.io/v1/datasets/polymarket/markets"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=paths.TELONEX_METADATA,
                    help=f"output path (default: {paths.TELONEX_METADATA})")
    ap.add_argument("--force", action="store_true",
                    help="redownload even if the file already exists")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        sz = args.out.stat().st_size / 1024 / 1024
        print(f"{args.out} already exists ({sz:.1f} MB); skipping. Use --force to redownload.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    print(f"  -> {args.out}")
    t0 = time.time()
    with requests.get(URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        n = 0
        last = 0.0
        with open(args.out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                n += len(chunk)
                now = time.time()
                if now - last > 2.0:
                    pct = (n / total * 100) if total else 0
                    rate = n / (now - t0 + 1e-9) / 1024 / 1024
                    print(f"  {n / 1024 / 1024:6.1f} MB"
                          + (f" / {total / 1024 / 1024:.1f} MB ({pct:.1f}%)" if total else "")
                          + f"  {rate:.1f} MB/s")
                    last = now
    print(f"\nwrote {args.out} ({n / 1024 / 1024:.1f} MB in {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
