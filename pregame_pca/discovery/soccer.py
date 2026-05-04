"""Soccer sport module: fetch markets from Gamma API and build SAT constraints."""

import json
import re
import urllib.request
from datetime import datetime, timezone, timedelta

from .complete_sets import ExactlyOne, ImplicationChain, Implies

from pregame_pca.polymarket.soccer import (
    SOCCER_SERIES_IDS,
    SPORTS_TYPE_MAP,
    parse_game_slug as _parse_game_slug,
)

GAMMA_API = "https://gamma-api.polymarket.com"


def _fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())



def fetch_markets(hours_ahead=48, hours_behind=3):
    """Fetch soccer markets from Gamma API.

    Returns list of {id, question, game, type, yes_token, no_token}.
    """
    now = datetime.now(timezone.utc)
    earliest = now - timedelta(hours=hours_behind)
    latest = now + timedelta(hours=hours_ahead)

    # Fetch events from all soccer series
    all_events = []
    seen_event_ids = set()
    for series_id in SOCCER_SERIES_IDS:
        offset = 0
        while True:
            url = (f"{GAMMA_API}/events?series_id={series_id}&closed=false&active=true"
                   f"&limit=200&offset={offset}")
            batch = _fetch_json(url)
            for event in batch:
                eid = event["id"]
                if eid not in seen_event_ids:
                    seen_event_ids.add(eid)
                    all_events.append(event)
            if len(batch) < 200:
                break
            offset += 200

    all_markets = []
    seen_ids = set()

    for event in all_events:
        # Filter by event startTime
        start_str = event.get("startTime")
        if not start_str:
            continue
        try:
            start_time = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        if start_time < earliest or start_time > latest:
            continue

        for market in event.get("markets", []):
            market_id = int(market["id"])
            if market_id in seen_ids:
                continue

            smt = market.get("sportsMarketType", "")
            market_type = SPORTS_TYPE_MAP.get(smt)
            if not market_type:
                continue

            raw_ids = market.get("clobTokenIds")
            if isinstance(raw_ids, str):
                raw_ids = json.loads(raw_ids)
            if not raw_ids or len(raw_ids) < 2:
                continue
            yes_token, no_token = raw_ids[0], raw_ids[1]

            game_slug = _parse_game_slug(event["slug"])

            seen_ids.add(market_id)
            all_markets.append({
                "id": market_id,
                "question": market["question"],
                "game": game_slug,
                "game_id": event.get("gameId"),
                "type": market_type,
                "yes_token": yes_token,
                "no_token": no_token,
                "start_time_ts": int(start_time.timestamp()),
            })

    # Sort by game then by type order for consistency
    type_order = {"moneyline": 0, "spread": 1, "totals": 2, "btts": 3}
    all_markets.sort(key=lambda m: (m["game"], type_order.get(m["type"], 99), m["id"]))

    return all_markets


def build_constraints(game_markets):
    """Build constraints for one game's markets.

    Args:
        game_markets: list of market dicts for a single game

    Returns:
        (n, constraints, label_fn, slot_info) where label_fn(tuple_of_literals) -> str
        and slot_info is a dict with keys: a_win_idx, b_win_idx, tie_idx, btts_idx
        (ints or None), spread_a, spread_b, ou (dict threshold -> index).
    """
    n = len(game_markets)

    # Parse market types
    moneylines = [m for m in game_markets if m["type"] == "moneyline"]
    wins = [m for m in moneylines if "win" in m["question"].lower()]
    ties = [m for m in moneylines if "draw" in m["question"].lower()]
    spreads = [m for m in game_markets if m["type"] == "spread"]
    totals = [m for m in game_markets if m["type"] == "totals"]
    btts_list = [m for m in game_markets if m["type"] == "btts"]

    if not ties:
        return None
    tie = ties[0]

    # Determine team names from vs. pattern
    vs_match = None
    for t in totals + btts_list:
        vs_match = re.search(r"(.+?) vs\. (.+?):", t["question"])
        if vs_match:
            break
    if not vs_match:
        vs_match = re.search(r"(.+?) vs\. (.+?)(?:\s+end\s)", tie["question"])
        if vs_match:
            team_a_raw = vs_match.group(1).strip()
            if team_a_raw.lower().startswith("will "):
                vs_match = re.search(r"Will (.+?) vs\. (.+?)(?:\s+end\s)", tie["question"])
    if not vs_match:
        return None

    team_a = vs_match.group(1).strip()
    team_b = vs_match.group(2).strip()

    # Map market types to indices
    a_win_idx = b_win_idx = tie_idx = btts_idx = None
    spread_a = {}
    spread_b = {}
    ou = {}

    for i, m in enumerate(game_markets):
        if m["type"] == "moneyline":
            if "draw" in m["question"].lower():
                tie_idx = i
            elif team_a in m["question"]:
                a_win_idx = i
            elif team_b in m["question"]:
                b_win_idx = i
        elif m["type"] == "spread":
            k_match = re.search(r"\(-([\d.]+)\)", m["question"])
            if k_match:
                thresh = float(k_match.group(1))
                if team_a in m["question"]:
                    spread_a[thresh] = i
                elif team_b in m["question"]:
                    spread_b[thresh] = i
        elif m["type"] == "totals":
            k_match = re.search(r"O/U ([\d.]+)", m["question"])
            if k_match:
                ou[float(k_match.group(1))] = i
        elif m["type"] == "btts":
            btts_idx = i

    # Build constraints
    constraints = []

    if a_win_idx is not None and b_win_idx is not None and tie_idx is not None:
        constraints.append(ExactlyOne([a_win_idx, b_win_idx, tie_idx]))

    # Team A spread chain: biggest => ... => smallest => a_win
    if spread_a:
        a_thresholds = sorted(spread_a.keys())
        a_chain = [spread_a[t] for t in reversed(a_thresholds)]
        if a_win_idx is not None:
            a_chain.append(a_win_idx)
        if len(a_chain) >= 2:
            constraints.append(ImplicationChain(a_chain))

    # Team B spread chain: biggest => ... => smallest => b_win
    if spread_b:
        b_thresholds = sorted(spread_b.keys())
        b_chain = [spread_b[t] for t in reversed(b_thresholds)]
        if b_win_idx is not None:
            b_chain.append(b_win_idx)
        if len(b_chain) >= 2:
            constraints.append(ImplicationChain(b_chain))

    for thresh, idx in spread_a.items():
        if thresh in ou:
            constraints.append(Implies(idx, ou[thresh]))
    for thresh, idx in spread_b.items():
        if thresh in ou:
            constraints.append(Implies(idx, ou[thresh]))

    # O/U chain: biggest => ... => smallest
    ou_thresholds = sorted(ou.keys())
    if len(ou_thresholds) >= 2:
        ou_chain = [ou[t] for t in reversed(ou_thresholds)]
        constraints.append(ImplicationChain(ou_chain))

    if btts_idx is not None and 1.5 in ou:
        constraints.append(Implies(btts_idx, ou[1.5]))

    # Build label function
    def label_fn(tup):
        spread_a_inv = {v: k for k, v in spread_a.items()}
        spread_b_inv = {v: k for k, v in spread_b.items()}
        ou_inv = {v: k for k, v in ou.items()}

        parts = []
        for lit in tup:
            idx = lit.index
            yes = lit.side
            if idx == a_win_idx:
                parts.append("A win" if yes else "NOT A win")
            elif idx == b_win_idx:
                parts.append("B win" if yes else "NOT B win")
            elif idx == tie_idx:
                parts.append("tie" if yes else "NO tie")
            elif idx in spread_a_inv:
                k = spread_a_inv[idx]
                parts.append(f"A+{k}" if yes else f"NOT A+{k}")
            elif idx in spread_b_inv:
                k = spread_b_inv[idx]
                parts.append(f"B+{k}" if yes else f"NOT B+{k}")
            elif idx in ou_inv:
                k = ou_inv[idx]
                parts.append(f"total>{k}" if yes else f"total<={k}")
            elif idx == btts_idx:
                parts.append("btts" if yes else "NO btts")
            else:
                m = game_markets[idx]
                side_str = "YES" if yes else "NO"
                parts.append(f"{m['question']} {side_str}")
        return ", ".join(parts)

    slot_info = {
        "a_win_idx": a_win_idx,
        "b_win_idx": b_win_idx,
        "tie_idx": tie_idx,
        "btts_idx": btts_idx,
        "spread_a": dict(spread_a),
        "spread_b": dict(spread_b),
        "ou": dict(ou),
    }

    return n, constraints, label_fn, slot_info
