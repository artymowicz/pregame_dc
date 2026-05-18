"""Standard slot map: convert soccer game's slot_info dict into the canonical
12-slot ordering used by the rank-3 PCR model.

Lifted verbatim from strategies/convex_arb/per_game_solver.py:standard_slot_map
in the source monorepo. Pure-Python; no numerical / OSQP dependencies.

Standard slot layout:
  0:ml_a, 1:ml_b, 2:ml_draw,
  3:spread_a1(1.5), 4:spread_b1(1.5), 5:spread_a2(2.5), 6:spread_b2(2.5),
  7:total_1(1.5), 8:total_2(2.5), 9:total_3(3.5), 10:total_4(4.5),
  11:btts
Only slots with a present market are included.
"""
from __future__ import annotations


def standard_slot_map(slot_info):
    """Convert soccer.build_constraints slot_info dict into
    {standard_slot: game_local_idx}. Missing slots are simply absent."""
    m = {}
    if slot_info["a_win_idx"] is not None:
        m[0] = slot_info["a_win_idx"]
    if slot_info["b_win_idx"] is not None:
        m[1] = slot_info["b_win_idx"]
    if slot_info["tie_idx"] is not None:
        m[2] = slot_info["tie_idx"]
    sa = slot_info["spread_a"]
    sb = slot_info["spread_b"]
    if 1.5 in sa:
        m[3] = sa[1.5]
    if 1.5 in sb:
        m[4] = sb[1.5]
    if 2.5 in sa:
        m[5] = sa[2.5]
    if 2.5 in sb:
        m[6] = sb[2.5]
    ou = slot_info["ou"]
    for slot_offset, thresh in enumerate([1.5, 2.5, 3.5, 4.5]):
        if thresh in ou:
            m[7 + slot_offset] = ou[thresh]
    if slot_info["btts_idx"] is not None:
        m[11] = slot_info["btts_idx"]
    return m
