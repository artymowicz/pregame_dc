"""Post-game backfill for the live bot's orders log.

Reads `orders_summary.jsonl`, queries Gamma for each game's resolved markets,
fills in outcome and realised_pnl, and writes the augmented log to
`orders_summary_resolved.jsonl` (the original is never modified).

Usage:
    python -m pregame_pca.live.resolve_outcomes
        [--input PATH] [--output PATH] [--min-age-hours 3]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

GAMMA = "https://gamma-api.polymarket.com"
from pregame_pca import paths
DEFAULT_INPUT = paths.ORDERS_SUMMARY
DEFAULT_OUTPUT = paths.ORDERS_RESOLVED


def _parse_outcome_prices(raw) -> list[str] | None:
    """Gamma returns outcomePrices as a JSON-encoded string like '["0", "1"]'."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        parsed = json.loads(raw)
        return [str(x) for x in parsed]
    except (json.JSONDecodeError, TypeError):
        return None


def fetch_market(market_id: int) -> dict | None:
    """Returns the Gamma market object or None on error."""
    try:
        r = requests.get(f"{GAMMA}/markets/{market_id}", timeout=15)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, json.JSONDecodeError):
        return None


def resolve_record(record: dict, market_cache: dict[int, dict | None],
                   min_age_seconds: float) -> dict:
    """Mutates a copy of `record` and returns it with outcome populated when
    the market is resolved."""
    out = dict(record)

    # Skip already-resolved or stub records (placed=False, dry-run, etc.)
    if out.get("outcome") is not None:
        return out
    if not out.get("placed"):
        return out
    if out.get("status") != "matched":
        # killed / error fills have no realised position
        return out

    # Skip if game might still be in progress
    start_ts = out.get("game_start_ts")
    if start_ts is not None and time.time() - float(start_ts) < min_age_seconds:
        return out

    market_id = out.get("market_id")
    if market_id is None:
        return out

    if market_id not in market_cache:
        market_cache[market_id] = fetch_market(int(market_id))
    market = market_cache[market_id]
    if market is None:
        return out
    if not market.get("closed"):
        return out

    prices = _parse_outcome_prices(market.get("outcomePrices"))
    if not prices or len(prices) < 2:
        return out

    side = out.get("side")  # "Y" or "N"
    yes_won = prices[0] == "1"
    no_won = prices[1] == "1"
    if side == "Y":
        outcome = 1 if yes_won else 0
    elif side == "N":
        outcome = 1 if no_won else 0
    else:
        return out

    out["outcome"] = outcome
    out["outcome_filled_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    filled_size = out.get("filled_size")
    notional = out.get("filled_notional_usdc")
    if filled_size is not None and notional is not None:
        out["realised_pnl"] = outcome * float(filled_size) - float(notional)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--min-age-hours", type=float, default=3.0,
                    help="Skip games whose start_ts is younger than this (still in progress)")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"input log not found: {args.input}")
        return

    min_age_s = args.min_age_hours * 3600
    market_cache: dict[int, dict | None] = {}

    n_total = 0
    n_resolved = 0
    n_already = 0
    n_in_progress = 0
    n_no_market = 0
    n_not_placed = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input) as f_in, open(args.output, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            n_total += 1

            if rec.get("outcome") is not None:
                n_already += 1
                f_out.write(json.dumps(rec, default=str) + "\n")
                continue
            if not rec.get("placed") or rec.get("status") != "matched":
                n_not_placed += 1
                f_out.write(json.dumps(rec, default=str) + "\n")
                continue

            updated = resolve_record(rec, market_cache, min_age_s)
            f_out.write(json.dumps(updated, default=str) + "\n")
            if updated.get("outcome") is not None:
                n_resolved += 1
            elif updated.get("game_start_ts") and time.time() - float(updated["game_start_ts"]) < min_age_s:
                n_in_progress += 1
            else:
                n_no_market += 1

    print(f"input:          {args.input}")
    print(f"output:         {args.output}")
    print(f"total records:  {n_total}")
    print(f"newly resolved: {n_resolved}")
    print(f"already resolved (passthrough): {n_already}")
    print(f"not-placed / killed: {n_not_placed}")
    print(f"still in progress (< {args.min_age_hours}h since start): {n_in_progress}")
    print(f"could not resolve via Gamma: {n_no_market}")

    # ---- Calibration sanity check ----
    # Re-read the just-written output (cleanest way to include all resolved
    # rows, including ones that were already resolved before this run).
    _calibration_check(args.output)


def _calibration_check(resolved_path):
    """Print four scalars over our resolved + matched fills:
        ω̂ = (1/N) Σ (o_i − p_i)         — model bias (signed model edge)
        SE_theory = √(Σ p_i(1−p_i)) / N — SE of ω̂ under H0 (model calibrated)
        SE_sample = √(Σ (o_i−p_i)²) / N — sample SE (no H0 assumption)
        ψ̂ = (1/N) Σ (o_i − q_i)         — realised average ¢/share PnL,
                                          where q_i is the actual fill price

    p_i is the model's predicted probability for the token we bought,
    o_i ∈ {0,1} is whether that token resolved YES,
    q_i = filled_notional / filled_size is the per-share price actually paid
    (uses FOK price-improvement when present).

    A z-score |Z| = |ω̂| / SE > 2 is grounds for suspicion that the model
    is mis-calibrated (under H0 you'd expect |Z| < 2 about 95% of the time).
    Sign flipped so positive numbers mean "the side we bought won more often
    than predicted/priced", which is the direction of trading edge.
    """
    import math
    from collections import defaultdict

    rows: list[tuple[str, float, int, float]] = []  # (market_type, p, o, q)
    with open(resolved_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("outcome") is None:
                continue
            if not rec.get("placed") or rec.get("status") != "matched":
                continue
            p = rec.get("predicted_prob_clipped")
            if p is None:
                p = rec.get("predicted_prob")
            if p is None:
                continue
            try:
                p = float(p)
                o = int(rec["outcome"])
            except (TypeError, ValueError):
                continue
            # q_i = price actually paid per share. Prefer notional/size to
            # capture FOK price improvement; fall back to the order limit.
            filled_size = rec.get("filled_size")
            filled_notional = rec.get("filled_notional_usdc")
            q = None
            if filled_size and filled_notional is not None:
                try:
                    q = float(filled_notional) / float(filled_size)
                except (TypeError, ValueError, ZeroDivisionError):
                    q = None
            if q is None:
                args = rec.get("order_args") or {}
                q = args.get("price")
                if q is None:
                    continue
                try:
                    q = float(q)
                except (TypeError, ValueError):
                    continue
            p_clipped = max(0.0, min(1.0, p))
            mtype = rec.get("market_type") or "unknown"
            rows.append((mtype, p_clipped, o, q))

    print()
    print(f"=== calibration sanity check (resolved + matched fills only) ===")
    if not rows:
        print("no resolved trades to test yet")
        return

    def _stats(subset):
        n = len(subset)
        ps = [r[1] for r in subset]
        os_ = [r[2] for r in subset]
        qs = [r[3] for r in subset]
        residuals = [o - p for p, o in zip(ps, os_)]   # o − p
        mu = sum(residuals) / n
        var_theory = sum(p * (1 - p) for p in ps)
        se_theory = math.sqrt(var_theory) / n if var_theory > 0 else 0.0
        var_sample = sum(r * r for r in residuals)
        se_sample = math.sqrt(var_sample) / n if var_sample > 0 else 0.0
        psi = sum(o - q for o, q in zip(os_, qs)) / n  # o − q (realised ¢/share)
        return n, mu, se_theory, se_sample, psi

    def _print_block(label, subset):
        n, mu, se_t, se_s, psi = _stats(subset)
        z_t = mu / se_t if se_t > 0 else 0.0
        z_s = mu / se_s if se_s > 0 else 0.0
        print(f"\n-- {label}  (N={n}) --")
        print(f"  ω̂  = (1/N) Σ (o_i − p_i)   = {mu:+.4f}    "
              f"(model bias: >0 ⇒ bought side wins more than model said)")
        print(f"  SE_theory = √(Σ p(1−p)) / N = {se_t:.4f}    "
              f"Z = {z_t:+.2f}    |Z| < 2: {'PASS' if abs(z_t) < 2 else 'FAIL'}")
        print(f"  SE_sample = √(Σ (o−p)²) / N = {se_s:.4f}    "
              f"Z = {z_s:+.2f}    |Z| < 2: {'PASS' if abs(z_s) < 2 else 'FAIL'}")
        print(f"  ψ̂  = (1/N) Σ (o_i − q_i)   = {psi:+.4f}    "
              f"(realised ¢/share PnL at fill price)")

    _print_block("ALL fills", rows)

    by_type: dict[str, list] = defaultdict(list)
    for r in rows:
        by_type[r[0]].append(r)
    for mtype in sorted(by_type):
        _print_block(f"market_type = {mtype}", by_type[mtype])


if __name__ == "__main__":
    main()
