"""Dataset quality filters for the Dome and self_collected soccer datasets.

Two independent quality checks:
  - Dome: counts unique per_game_data timestamps in the [-30min, +180min] game
    window. Callers pick their own `min_ts` cutoff inline (density varies
    sharply with game age, so there's no single right answer). The helper
    `compute_dome_coverage` returns the per-game table; `filter_dome` is a
    diagnostic wrapper that also writes a histogram.
  - self_collected: drops games whose window isn't fully covered by a continuous
    ws_logger session (built per-market; tolerates the midnight auto-
    rotation but not genuine logger restarts). Pre-computed into
    `data/self_collected/games_filtered.csv` because session coverage is orthogonal to
    per_game_data density and genuinely useful as a persistent filter.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRE_MS = 30 * 60_000
POST_MS = 180 * 60_000
SESSION_MERGE_GAP_MS = 60_000  # < 60s gap = midnight rotation, not a real outage
DEFAULT_MIN_DOME_TS = 100      # histogram valley between the broken-game tail and the main mode


# ---------------------------------------------------------------------------
# Dome
# ---------------------------------------------------------------------------

def compute_dome_coverage(dome_dir: Path | str) -> pd.DataFrame:
    """Per-game Dome per_game_data coverage in the [-30min, +180min] game window.

    Reads `<dome_dir>/games.csv` and the parquet files under
    `<dome_dir>/per_game_data/`. Returns one row per game_slug (games without a
    parquet or with an unparseable game_start_time are skipped).

    Columns:
      game_slug            : str
      n_rows_window        : int   — raw per_game_data rows inside the window
      n_unique_ts_window   : int   — unique timestamp_ms inside the window
      n_markets            : int   — unique market_ids across the whole parquet
    """
    dome_dir = Path(dome_dir)
    games = pd.read_csv(dome_dir / "games.csv")
    gst = games.groupby("game_slug")["game_start_time"].first().to_dict()
    snap_dir = dome_dir / "per_game_data"

    rows = []
    for slug, gst_s in gst.items():
        p = snap_dir / f"{slug}.parquet"
        if not p.exists():
            continue
        try:
            gst_ms = int(pd.Timestamp(gst_s).timestamp() * 1000)
        except Exception:
            continue
        win_start = gst_ms - PRE_MS
        win_end = gst_ms + POST_MS
        df = pd.read_parquet(p, columns=["timestamp_ms", "market_id"])
        win = df[(df.timestamp_ms >= win_start) & (df.timestamp_ms <= win_end)]
        rows.append({
            "game_slug": slug,
            "n_rows_window": len(win),
            "n_unique_ts_window": win.timestamp_ms.nunique() if len(win) else 0,
            "n_markets": df.market_id.nunique(),
        })

    return pd.DataFrame(rows)


def filter_dome(
    dome_dir: Path | str,
    out_dir: Path | str,
    min_ts: int = DEFAULT_MIN_DOME_TS,
) -> pd.DataFrame:
    """Diagnostic coverage report for the Dome per_game_data set.

    Writes:
      <out_dir>/dome_coverage.csv           — per-game row/per_game_data counts
      <out_dir>/dome_coverage_histogram.png — unique-ts histogram

    `min_ts` is used only to annotate the histogram and for the summary
    print. No games_filtered.csv is written — consumers apply their own
    `min_ts` inline via `compute_dome_coverage`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cov = compute_dome_coverage(dome_dir)
    cov["keep"] = cov.n_unique_ts_window >= min_ts
    cov.to_csv(out_dir / "dome_coverage.csv", index=False)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    bins = np.logspace(0, np.log10(max(cov.n_unique_ts_window.max(), 2) + 1), 50)
    ax.hist(cov.n_unique_ts_window.clip(lower=1), bins=bins)
    ax.set_xscale("log")
    for c, color in [(50, "red"), (100, "orange"), (200, "green")]:
        ax.axvline(c, color=color, ls="--", label=f"cutoff={c}")
    ax.set_xlabel("unique per_game_data timestamps in [-30, +180] game window")
    ax.set_ylabel("# games")
    ax.set_title(f"Dome per_game_data coverage per game (n={len(cov)})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "dome_coverage_histogram.png", dpi=110)
    plt.close()

    print(f"[filter-dome] {len(cov)} games scanned, "
          f"keep {cov.keep.sum()} (>= {min_ts} ts), drop {(~cov.keep).sum()}")
    return cov


# ---------------------------------------------------------------------------
# Dome weather (daily-temperature events)
# ---------------------------------------------------------------------------

def compute_dome_weather_coverage(data_dir: Path | str) -> pd.DataFrame:
    """Per-event Dome per_game_data coverage over the full event lifetime.

    Reads `<data_dir>/events.csv` and the parquet files under
    `<data_dir>/per_game_data/`. Returns one row per event_key (events without a
    parquet or with unparseable start/end times are skipped).

    Columns:
      event_key            : str
      n_rows_window        : int   — raw per_game_data rows inside the event window
      n_unique_ts_window   : int   — unique timestamp_ms inside the window
      n_markets            : int   — unique market_ids across the whole parquet
    """
    data_dir = Path(data_dir)
    events = pd.read_csv(data_dir / "events.csv")
    windows = events.groupby("event_key").agg(
        start=("event_start_time", "first"),
        end=("event_end_time", "first"),
    ).to_dict("index")
    snap_dir = data_dir / "per_game_data"

    rows = []
    for key, w in windows.items():
        p = snap_dir / f"{key}.parquet"
        if not p.exists():
            continue
        try:
            win_start = int(pd.Timestamp(w["start"]).timestamp() * 1000)
            win_end = int(pd.Timestamp(w["end"]).timestamp() * 1000)
        except Exception:
            continue
        df = pd.read_parquet(p, columns=["timestamp_ms", "market_id"])
        win = df[(df.timestamp_ms >= win_start) & (df.timestamp_ms <= win_end)]
        rows.append({
            "event_key": key,
            "n_rows_window": len(win),
            "n_unique_ts_window": win.timestamp_ms.nunique() if len(win) else 0,
            "n_markets": df.market_id.nunique(),
        })

    return pd.DataFrame(rows)


def filter_dome_weather(
    data_dir: Path | str,
    out_dir: Path | str,
    min_ts: int = DEFAULT_MIN_DOME_TS,
) -> pd.DataFrame:
    """Diagnostic coverage report for the Dome weather per_game_data set.

    Writes:
      <out_dir>/dome_weather_coverage.csv           — per-event row/per_game_data counts
      <out_dir>/dome_weather_coverage_histogram.png — unique-ts histogram

    `min_ts` is used only to annotate the histogram and for the summary
    print. No events_filtered.csv is written — consumers apply their own
    `min_ts` inline via `compute_dome_weather_coverage`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cov = compute_dome_weather_coverage(data_dir)
    cov["keep"] = cov.n_unique_ts_window >= min_ts
    cov.to_csv(out_dir / "dome_weather_coverage.csv", index=False)

    if len(cov):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        bins = np.logspace(0, np.log10(max(cov.n_unique_ts_window.max(), 2) + 1), 50)
        ax.hist(cov.n_unique_ts_window.clip(lower=1), bins=bins)
        ax.set_xscale("log")
        for c, color in [(50, "red"), (100, "orange"), (200, "green")]:
            ax.axvline(c, color=color, ls="--", label=f"cutoff={c}")
        ax.set_xlabel("unique per_game_data timestamps in event lifetime")
        ax.set_ylabel("# events")
        ax.set_title(f"Dome weather per_game_data coverage per event (n={len(cov)})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "dome_weather_coverage_histogram.png", dpi=110)
        plt.close()

    print(f"[filter-dome-weather] {len(cov)} events scanned, "
          f"keep {int(cov.keep.sum()) if len(cov) else 0} (>= {min_ts} ts), "
          f"drop {int((~cov.keep).sum()) if len(cov) else 0}")
    return cov


# ---------------------------------------------------------------------------
# self_collected
# ---------------------------------------------------------------------------

def _scan_self_collected_files(self_collected_log_dir: Path):
    infos, corrupt = [], []
    for fp in sorted(self_collected_log_dir.glob("ws_log_soccer_*.h5")):
        try:
            with h5py.File(fp, "r") as f:
                meta_raw = [
                    m.decode() if isinstance(m, bytes) else m
                    for m in f["/meta/markets"][:]
                ]
                meta = set(m for m in meta_raw if m)
                pc = f["/market/price_change"]
                n = pc["timestamp"].shape[0]
                if n == 0 or not meta:
                    continue
                t_min = int(pc["timestamp"][0])
                t_max = int(pc["timestamp"][-1])
            infos.append({
                "name": fp.name,
                "t_min": t_min,
                "t_max": t_max,
                "markets": meta,
            })
        except Exception as e:
            corrupt.append((fp.name, str(e)))
    return infos, corrupt


def _per_market_sessions(file_infos):
    per_market: dict[str, list[tuple[int, int]]] = {}
    for fi in file_infos:
        for mid in fi["markets"]:
            per_market.setdefault(mid, []).append((fi["t_min"], fi["t_max"]))
    out = {}
    for mid, ivs in per_market.items():
        ivs.sort()
        merged = []
        for a, b in ivs:
            if merged and a - merged[-1][1] < SESSION_MERGE_GAP_MS:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        out[mid] = merged
    return out


def filter_self_collected(
    self_collected_log_dir: Path | str,
    self_collected_data_dir: Path | str,
    out_dir: Path | str,
) -> pd.DataFrame:
    """Drop self_collected games whose window isn't fully covered by one ws_logger session.

    Writes:
      <self_collected_data_dir>/games_filtered.csv  — filtered games.csv subset
      <out_dir>/self_collected_sessions.csv         — per-file time ranges
      <out_dir>/self_collected_coverage.csv         — per-game coverage status
      <out_dir>/self_collected_logger_corrupt.txt   — unreadable self_collected files (if any)
    """
    self_collected_log_dir = Path(self_collected_log_dir)
    self_collected_data_dir = Path(self_collected_data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_infos, corrupt = _scan_self_collected_files(self_collected_log_dir)
    print(f"[filter-self_collected] {len(file_infos)} usable ws_log files, {len(corrupt)} corrupt")
    if corrupt:
        with open(out_dir / "self_collected_logger_corrupt.txt", "w") as f:
            for name, err in corrupt:
                f.write(f"{name}\t{err}\n")
                print(f"            CORRUPT {name}: {err}")

    pd.DataFrame([{
        "name": fi["name"],
        "t_min": fi["t_min"],
        "t_max": fi["t_max"],
        "n_markets": len(fi["markets"]),
        "duration_h": (fi["t_max"] - fi["t_min"]) / 3_600_000,
    } for fi in file_infos]).to_csv(out_dir / "self_collected_sessions.csv", index=False)

    per_market = _per_market_sessions(file_infos)

    games = pd.read_csv(self_collected_data_dir / "games.csv")
    game_markets = games.groupby("game_slug")["market_id"].apply(
        lambda s: [str(m) for m in s]).to_dict()
    game_start = games.groupby("game_slug")["game_start_time"].first().to_dict()
    parquets_present = {p.stem for p in (self_collected_data_dir / "per_game_data").glob("*.parquet")}

    results = []
    for slug, mids in game_markets.items():
        try:
            gst_ms = int(pd.Timestamp(game_start[slug]).timestamp() * 1000)
        except Exception:
            continue
        win_start = gst_ms - PRE_MS
        win_end = gst_ms + POST_MS

        per_mkt_overlap = []
        per_mkt_full = []
        for mid in mids:
            ivs = per_market.get(mid, [])
            best = 0
            full = False
            for a, b in ivs:
                ov = max(0, min(b, win_end) - max(a, win_start))
                if ov > best:
                    best = ov
                if a <= win_start and b >= win_end:
                    full = True
                    break
            per_mkt_overlap.append(best)
            per_mkt_full.append(full)

        min_overlap = min(per_mkt_overlap) if per_mkt_overlap else 0
        all_full = bool(per_mkt_full and all(per_mkt_full))

        results.append({
            "game_slug": slug,
            "game_start": game_start[slug],
            "n_markets": len(mids),
            "min_overlap_min": min_overlap / 60_000,
            "covered": all_full,
            "has_parquet": slug in parquets_present,
        })

    cov = pd.DataFrame(results)
    cov.to_csv(out_dir / "self_collected_coverage.csv", index=False)

    keep_set = set(cov.loc[cov.covered, "game_slug"])
    games[games.game_slug.isin(keep_set)].to_csv(
        self_collected_data_dir / "games_filtered.csv", index=False)

    n_p = int(cov.has_parquet.sum())
    n_p_cov = int(((cov.has_parquet) & (cov.covered)).sum())
    print(f"[filter-self_collected] {len(cov)} games in csv: covered={cov.covered.sum()}, "
          f"partial={(~cov.covered).sum()}")
    print(f"            of {n_p} with parquets: covered={n_p_cov}, "
          f"partial={n_p - n_p_cov}")
    return cov
