"""Canonical constants for the rank-3 PCR pregame model.

The 12 markets per soccer game are ordered as:

    0:  A win        moneyline
    1:  B win        moneyline
    2:  Draw         moneyline
    3:  A -1.5       spread
    4:  B -1.5       spread
    5:  A -2.5       spread
    6:  B -2.5       spread
    7:  Over 1.5     totals
    8:  Over 2.5     totals
    9:  Over 3.5     totals
    10: Over 4.5     totals
    11: BTTS         both teams to score

The labeled datasets carry one ask per side per market in 24 columns:

    x_0..x_11   YES asks  (slots 0..11)
    x_12..x_23  NO  asks  (slots 0..11, NO side)

Outcomes are stored as 12 binary columns y_0..y_11 (the YES-side outcome of
each canonical slot). The 24-element outcome vector consumed by the model is
reconstructed via  Y = hstack([y, 1 - y]).
"""
from __future__ import annotations

MARKET_LABELS: list[str] = [
    "A win", "B win", "Draw",
    "A -1.5", "B -1.5", "A -2.5", "B -2.5",
    "O 1.5", "O 2.5", "O 3.5", "O 4.5",
    "BTTS",
]

# YES + NO side labels in canonical 24-token order, useful for plot legends.
LABELS_24: list[str] = (
    [f"{l} YES ask" for l in MARKET_LABELS]
    + [f"{l} NO ask" for l in MARKET_LABELS]
)

X_COLS: list[str] = [f"x_{i}" for i in range(24)]   # 24 ask columns
S_COLS: list[str] = [f"s_{i}" for i in range(24)]   # 24 size columns
Y_COLS: list[str] = [f"y_{i}" for i in range(12)]   # 12 outcome columns

TYPE_FOR_SLOT: dict[int, str] = {
    0: "moneyline", 1: "moneyline", 2: "moneyline",
    3: "spread", 4: "spread", 5: "spread", 6: "spread",
    7: "totals", 8: "totals", 9: "totals", 10: "totals",
    11: "btts",
}
