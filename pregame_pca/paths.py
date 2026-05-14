"""Central path constants for pregame_pca.

Resolved relative to the package directory at import time. The directory
layout is:

    PACKAGE_ROOT/
        pregame_pca/         <- this file lives here
        data/
            labeled/
            self_collected/
            telonex/
        logs/
"""
from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# ---- data ----------------------------------------------------------
DATA_DIR        = PACKAGE_ROOT / "data"
LABELED_DIR     = DATA_DIR / "labeled"
OUTCOMES        = DATA_DIR / "outcomes.parquet"
TELONEX_LABELED = LABELED_DIR / "telonex_dataset.parquet"
SELF_COLLECTED_LABELED      = LABELED_DIR / "self_collected_dataset.parquet"

SELF_COLLECTED_DIR          = DATA_DIR / "self_collected"
SELF_COLLECTED_GAMES_CSV    = SELF_COLLECTED_DIR / "games.csv"
SELF_COLLECTED_SNAPSHOTS    = SELF_COLLECTED_DIR / "per_game_data"

TELONEX_DIR     = DATA_DIR / "telonex"
TELONEX_GAMES_CSV = TELONEX_DIR / "games.csv"
TELONEX_SNAPSHOTS = TELONEX_DIR / "per_game_data"
TELONEX_METADATA = TELONEX_DIR / "metadata" / "markets.parquet"
TELONEX_KICKOFF_CACHE = TELONEX_DIR / "cache" / "kickoff_times.json"

# ---- models --------------------------------------------------------
MODELS_DIR      = PACKAGE_ROOT / "pregame_pca" / "models"
MODEL_T_25MIN   = MODELS_DIR / "rank3_t-25min.npz"
MODEL_T_10MIN   = MODELS_DIR / "rank3_t-10min.npz"
MODEL_T_25MIN_IMP = MODELS_DIR / "rank3_t-25min_imp.npz"
MODEL_T_10MIN_IMP = MODELS_DIR / "rank3_t-10min_imp.npz"

# ---- runtime / live bot logs --------------------------------------
LOGS_DIR        = PACKAGE_ROOT / "logs"
WS_EVENTS_DIR   = LOGS_DIR / "ws_events"
WS_MASTER_LOG   = LOGS_DIR / "ws_events_master.jsonl"
ORDERS_SUMMARY  = LOGS_DIR / "orders_summary.jsonl"
ORDERS_RESOLVED = LOGS_DIR / "orders_summary_resolved.jsonl"
KICKOFFS_LOG    = LOGS_DIR / "kickoffs.jsonl"
STATE_FILE      = LOGS_DIR / "state.json"

# ---- data_collection paths ---------------------------------------
RAW_WS_LOGS     = DATA_DIR / "raw_ws_logs"
MARKETS_DIR     = PACKAGE_ROOT / "data_collection" / "markets"
MARKETS_CSV     = MARKETS_DIR / "soccer.csv"


def source_games_csv(source: str) -> Path:
    if source == "self_collected":
        return SELF_COLLECTED_GAMES_CSV
    if source == "telonex":
        return TELONEX_GAMES_CSV
    raise ValueError(f"unknown source: {source!r} (expected 'self_collected' or 'telonex')")


def source_per_game_data_dir(source: str) -> Path:
    if source == "self_collected":
        return SELF_COLLECTED_SNAPSHOTS
    if source == "telonex":
        return TELONEX_SNAPSHOTS
    raise ValueError(f"unknown source: {source!r} (expected 'self_collected' or 'telonex')")
