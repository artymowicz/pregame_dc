"""General-purpose constraint-satisfaction complete set finder for sports markets."""

from itertools import combinations, product
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# --- Constraint types ---

@dataclass
class ExactlyOne:
    """Exactly one of these market indices resolves YES."""
    indices: list

@dataclass
class AtMostOne:
    """At most one of these market indices resolves YES."""
    indices: list

@dataclass
class AtLeastOne:
    """At least one of these market indices resolves YES."""
    indices: list

@dataclass
class Implies:
    """If market a resolves YES, then market b resolves YES."""
    a: int
    b: int

@dataclass
class ImplicationChain:
    """Ordered chain: indices[0] => indices[1] => ... => indices[-1].
    Has len(indices)+1 valid assignments (threshold levels)."""
    indices: list

@dataclass
class Literal:
    """A market-side literal: market[index] == side."""
    index: int
    side: bool  # True = YES, False = NO

    def __repr__(self):
        return f"Literal({self.index}, {'YES' if self.side else 'NO'})"


def _build_worlds(n, constraints):
    """Build valid worlds by exploiting ExactlyOne/AtMostOne structure.

    Instead of enumerating all 2^n worlds then filtering, construct valid
    worlds directly from constraint assignments, dramatically reducing
    the search space.
    """
    group_cs = [c for c in constraints if isinstance(c, (ExactlyOne, AtMostOne, ImplicationChain))]
    implies_cs = [c for c in constraints if isinstance(c, Implies)]

    # S = all indices appearing in group constraints
    s_indices = set()
    for gc in group_cs:
        s_indices.update(gc.indices)
    free_indices = sorted(set(range(n)) - s_indices)
    s_indices_sorted = sorted(s_indices)

    # Generate per-constraint valid assignments (list of lists of dicts)
    per_constraint = []
    for gc in group_cs:
        assignments = []
        if isinstance(gc, ExactlyOne):
            for chosen in gc.indices:
                assignments.append({i: (i == chosen) for i in gc.indices})
        elif isinstance(gc, AtMostOne):
            assignments.append({i: False for i in gc.indices})
            for chosen in gc.indices:
                assignments.append({i: (i == chosen) for i in gc.indices})
        elif isinstance(gc, ImplicationChain):
            m = len(gc.indices)
            # Level 0: all NO
            assignments.append({idx: False for idx in gc.indices})
            # Level k: indices[m-k:] are YES, indices[:m-k] are NO
            for k in range(1, m + 1):
                asn = {}
                for j, idx in enumerate(gc.indices):
                    asn[idx] = (j >= m - k)
                assignments.append(asn)
        per_constraint.append(assignments)

    # Cartesian product with conflict detection
    valid_s_assignments = [{}]
    for assignments in per_constraint:
        next_valid = []
        for existing in valid_s_assignments:
            for new_asn in assignments:
                merged = existing.copy()
                conflict = False
                for idx, val in new_asn.items():
                    if idx in merged and merged[idx] != val:
                        conflict = True
                        break
                    merged[idx] = val
                if not conflict:
                    next_valid.append(merged)
        valid_s_assignments = next_valid

    # Build S-columns: one row per valid S-assignment
    num_s = len(valid_s_assignments)
    num_free = len(free_indices)
    num_free_worlds = 1 << num_free

    if num_s == 0:
        return np.empty((0, n), dtype=bool)

    # Build S-assignment matrix (num_s, n) — only S columns filled, free columns set later
    s_rows = np.zeros((num_s, n), dtype=bool)
    for r, asn in enumerate(valid_s_assignments):
        for idx, val in asn.items():
            s_rows[r, idx] = val

    # Build free-columns matrix (num_free_worlds, n) — only free columns filled
    free_rows = np.zeros((num_free_worlds, n), dtype=bool)
    for w in range(num_free_worlds):
        for bit, fi in enumerate(free_indices):
            if (w >> bit) & 1:
                free_rows[w, fi] = True

    # Cartesian product: repeat each S-row for every free combination
    # s_rows[i] combined with free_rows[j] for all i,j
    omega = np.repeat(s_rows, num_free_worlds, axis=0)
    free_tiled = np.tile(free_rows, (num_s, 1))
    # Merge: S columns come from s_rows, free columns from free_rows
    for fi in free_indices:
        omega[:, fi] = free_tiled[:, fi]

    # Filter by Implies constraints
    if implies_cs:
        mask = np.ones(len(omega), dtype=bool)
        for c in implies_cs:
            mask &= ~omega[:, c.a] | omega[:, c.b]
        omega = omega[mask]

    # Filter by AtLeastOne constraints
    atleastone_cs = [c for c in constraints if isinstance(c, AtLeastOne)]
    if atleastone_cs:
        mask = np.ones(len(omega), dtype=bool)
        for c in atleastone_cs:
            covered = np.zeros(len(omega), dtype=bool)
            for idx in c.indices:
                covered |= omega[:, idx]
            mask &= covered
        omega = omega[mask]

    return omega


def computeCompleteSets(n, constraints, k=3):
    """Find all minimal complete tuples up to size k.

    Args:
        n: number of markets
        constraints: list of ExactlyOne / AtMostOne / Implies constraint objects
        k: max tuple size (default 3)

    Returns:
        list of minimal complete tuples, each a list of Literal objects
    """
    # Step 1: Build valid worlds (Omega) via constraint-aware generation
    omega = _build_worlds(n, constraints)

    if len(omega) == 0:
        return []

    # Step 2: Enumerate candidate literals (2n total: each market x 2 sides)
    all_literals = []
    for i in range(n):
        all_literals.append(Literal(i, True))
        all_literals.append(Literal(i, False))

    # Precompute satisfaction vectors: for each literal, which worlds it satisfies
    # sat[j] is a boolean array of shape (|Omega|,)
    sat = np.empty((len(all_literals), len(omega)), dtype=bool)
    for j, lit in enumerate(all_literals):
        if lit.side:
            sat[j] = omega[:, lit.index]
        else:
            sat[j] = ~omega[:, lit.index]

    # Step 3: Find complete tuples of size 1..k
    complete_by_size = {l: [] for l in range(1, k + 1)}
    # Track which single-literal indices are complete (for minimality filtering)
    complete_singles = set()
    # Track which pairs are complete (for size-3 minimality)
    complete_pairs = set()

    for l in range(1, k + 1):
        for combo in combinations(range(len(all_literals)), l):
            # Skip if combo contains both YES and NO for same market
            markets_in_combo = {}
            skip = False
            for j in combo:
                lit = all_literals[j]
                if lit.index in markets_in_combo:
                    skip = True
                    break
                markets_in_combo[lit.index] = lit.side
            if skip:
                continue

            # Check completeness: every world must satisfy at least one literal
            covered = np.zeros(len(omega), dtype=bool)
            for j in combo:
                covered |= sat[j]
            if not covered.all():
                continue

            # Minimality check
            if l == 1:
                complete_singles.add(combo[0])
                complete_by_size[l].append([all_literals[j] for j in combo])
            elif l == 2:
                # Minimal if neither single literal alone is complete
                if combo[0] in complete_singles or combo[1] in complete_singles:
                    continue
                complete_pairs.add(combo)
                complete_by_size[l].append([all_literals[j] for j in combo])
            elif l == 3:
                # Minimal if no single is complete and no pair subset is complete
                if any(j in complete_singles for j in combo):
                    continue
                has_complete_pair = False
                for pair in combinations(combo, 2):
                    if pair in complete_pairs:
                        has_complete_pair = True
                        break
                if has_complete_pair:
                    continue
                complete_by_size[l].append([all_literals[j] for j in combo])

    result = []
    for l in range(1, k + 1):
        result.extend(complete_by_size[l])
    return result


