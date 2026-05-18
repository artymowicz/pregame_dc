"""Helpers for the labeled-dataset builder.

Consolidates utilities previously scattered across:
  - scripts/build_self_collected_pmxt_finetune_dataset.py:{_collect_games, split_by_hash,
                                                PRE_GAME_MIN, POST_GAME_MIN}
  - scripts/build_mlp_v2_dataset.py:{pivot_per_game_data, canonical_slot_market_ids,
                                     world_outcomes_for_game}
  - scripts/build_pretrain_dataset.py:{_load_per_game_data}

Outcomes are emitted directly as 12 binary y_k columns (the YES-side outcome
of each canonical slot), bypassing the world_idx intermediate that the source
repo uses. This keeps the labeled-dataset rebuild self-contained without
shipping V24.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from pregame_dc import paths
from pregame_dc.discovery import soccer as soccer_domain
from pregame_dc.discovery.slots import standard_slot_map

PRE_GAME_MIN = 30
POST_GAME_MIN = 180


def split_by_hash(slug: str, train_frac: float) -> str:
    h = int(hashlib.md5(slug.encode()).hexdigest(), 16) % 100
    return "train" if h < int(train_frac * 100) else "val"


def canonical_slot_market_ids(game_markets) -> list[int] | None:
    """Return [market_id_for_slot_0, ..., market_id_for_slot_11] or None when
    the game doesn't have all 12 canonical slots."""
    type_order = {"moneyline": 0, "spread": 1, "totals": 2, "btts": 3}
    for m in game_markets:
        m["id"] = int(m["market_id"])
    sorted_markets = sorted(
        game_markets, key=lambda m: (type_order.get(m["type"], 99), m["id"]),
    )
    result = soccer_domain.build_constraints(sorted_markets)
    if result is None:
        return None
    _, _, _, slot_info = result
    slot_map = standard_slot_map(slot_info)
    if len(slot_map) != 12:
        return None
    return [sorted_markets[slot_map[s]]["id"] for s in range(12)]


def slot_outcomes_for_game(slot_mids: list[int], outcome_map: dict[int, float]
                           ) -> list[int] | None:
    """Return [y_0, ..., y_11] in {0, 1}^12 from the canonical slot market_ids
    and a market_id → final_price (0 or 1) map. Returns None if any market is
    unresolved or carries a non-binary final price."""
    out = []
    for m in slot_mids:
        fp = outcome_map.get(int(m))
        if fp is None or pd.isna(fp):
            return None
        v = int(round(float(fp)))
        if v not in (0, 1):
            return None
        out.append(v)
    return out


def collect_games(games_csv: Path) -> dict[str, dict]:
    """Return {slug: {slot_mids, slot_outcomes, game_start_ts, games_subset}}.

    Only games with all 12 canonical slots AND fully resolved outcomes are
    kept. `slot_outcomes` is a 12-element list of YES-side outcomes (0/1),
    one per canonical slot.
    """
    games_df = pd.read_csv(games_csv)
    games_df["market_id"] = games_df["market_id"].astype("int64")
    # Cast to second-precision datetime64 first so this works across the
    # pandas 2.x (nanosecond default) / 3.x (microsecond default) split.
    # Use .timestamp() so seconds-since-epoch is unambiguous across the
    # pandas 2.x / 3.x default-unit change.
    games_df["game_start_ts"] = (
        pd.to_datetime(games_df["game_start_time"], utc=True)
          .map(lambda ts: int(ts.timestamp()))
          .astype("int64")
    )
    outcomes = pd.read_parquet(paths.OUTCOMES)
    outcomes["market_id"] = outcomes["market_id"].astype("int64")
    out_map = dict(zip(outcomes["market_id"], outcomes["final_price"]))

    kept: dict[str, dict] = {}
    for slug, grp in games_df.groupby("game_slug"):
        mids = canonical_slot_market_ids(grp.to_dict("records"))
        if mids is None:
            continue
        ys = slot_outcomes_for_game(mids, out_map)
        if ys is None:
            continue
        kept[slug] = {
            "slot_mids": mids,
            "slot_outcomes": ys,
            "game_start_ts": int(grp["game_start_ts"].iloc[0]),
            "games_subset": grp,
        }
    return kept


def load_per_game_data(source: str, snap_path: Path) -> pd.DataFrame | None:
    """Return a DataFrame with columns:
        market_id (int64), is_yes (bool), timestamp_ms (int64),
        best_ask (float), best_ask_size (float).
    Returns None if the per_game_data is missing required columns."""
    if source not in ("self_collected", "telonex"):
        raise ValueError(f"unknown source: {source!r}")
    df = pd.read_parquet(snap_path)
    needed = ["market_id", "is_yes", "timestamp_ms", "best_ask", "best_ask_size"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return None
    return df[needed]


def _pivot_one(snap_df, slot_market_ids, value_col, fill_value):
    wide = snap_df.pivot_table(
        index="timestamp_ms",
        columns=["market_id", "is_yes"],
        values=value_col,
        aggfunc="last",
    ).sort_index()

    cols = []
    for is_yes in (True, False):
        for mid in slot_market_ids:
            col = (mid, is_yes)
            if col not in wide.columns:
                wide[col] = np.nan
            cols.append(col)
    return wide[cols].ffill().fillna(fill_value)


def pivot_per_game_data(snap_df, slot_market_ids, n_samples=0, bucket_s=30,
                    grid_start_ms=None, grid_end_ms=None):
    """Pivot a game's per_game_data parquet into canonical 24-token wide format
    with fixed-grid 30s sampling over [grid_start_ms, grid_end_ms]."""
    snap_df = snap_df.sort_values("timestamp_ms")

    x = _pivot_one(snap_df, slot_market_ids, "best_ask", fill_value=1.0)
    x.columns = [f"x_{i}" for i in range(24)]
    s = _pivot_one(snap_df, slot_market_ids, "best_ask_size", fill_value=0.0)
    s.columns = [f"s_{i}" for i in range(24)]
    wide = pd.concat([x, s], axis=1)

    if bucket_s > 0:
        bucket_ms = int(bucket_s * 1000)
        if grid_start_ms is not None and grid_end_ms is not None:
            grid = np.arange(int(grid_start_ms),
                             int(grid_end_ms) + bucket_ms,
                             bucket_ms, dtype=np.int64)
            cols = [f"x_{i}" for i in range(24)] + [f"s_{i}" for i in range(24)]
            arr = np.empty((len(grid), 48), dtype=np.float32)
            arr[:, :24] = 1.0
            arr[:, 24:] = 0.0
            if len(wide) > 0:
                wide_idx = wide.index.values.astype(np.int64)
                pos = np.searchsorted(wide_idx, grid, side="right") - 1
                valid = pos >= 0
                if valid.any():
                    arr[valid] = wide.values[pos[valid]]
            wide = pd.DataFrame(arr, index=pd.Index(grid, name="timestamp_ms"),
                                columns=cols)
        elif len(wide) > 0:
            bucket_ids = (wide.index.values // bucket_ms).astype("int64")
            mask = np.concatenate([bucket_ids[:-1] != bucket_ids[1:], [True]])
            wide = wide[mask]

    if n_samples > 0 and len(wide) > n_samples:
        idx = np.linspace(0, len(wide) - 1, n_samples).round().astype(int)
        wide = wide.iloc[idx]

    return wide.reset_index()
