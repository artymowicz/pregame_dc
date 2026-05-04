"""Discover soccer markets and write the CSV that ws_logger consumes.

Trimmed replacement for strategies.arb_monitor.discovery.run_domain in the
source monorepo, restricted to the single domain pregame_pca cares about
(soccer). Reuses pregame_pca.discovery.soccer.fetch_markets so we don't have
to carry over the multi-sport arb_monitor.discovery module.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pregame_pca.discovery.soccer import fetch_markets
from pregame_pca import paths


CSV_FIELDS = ["id", "question", "game", "game_id", "type",
              "yes_token", "no_token"]


def discover_soccer_markets(out_csv: Path = None, hours_ahead: int = 48) -> int:
    """Fetch soccer markets from Gamma and write a CSV in the schema
    ws_logger.MarketRegistry expects. Returns the number of markets written."""
    out_csv = Path(out_csv) if out_csv is not None else paths.MARKETS_CSV
    markets = fetch_markets(hours_ahead=hours_ahead)
    # Propagate game_id from the main event to the -more-markets event
    # (which lacks gameId from Gamma). Same logic as arb_monitor's run_domain.
    slug_to_gid = {m["game"]: m["game_id"] for m in markets if m.get("game_id")}
    for m in markets:
        if not m.get("game_id"):
            m["game_id"] = slug_to_gid.get(m["game"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(markets)
    return len(markets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None,
                    help=f"output CSV path (default {paths.MARKETS_CSV})")
    ap.add_argument("--hours-ahead", type=int, default=48)
    args = ap.parse_args()
    n = discover_soccer_markets(args.out, hours_ahead=args.hours_ahead)
    print(f"wrote {n} markets to {args.out or paths.MARKETS_CSV}")


if __name__ == "__main__":
    main()
