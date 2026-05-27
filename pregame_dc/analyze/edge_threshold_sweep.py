"""5-fold cross-validated edge-rule threshold sweep for Dixon-Coles.

Splits the telonex labeled dataset (default
``data/labeled/telonex_dataset.parquet``) into 5 folds by game_slug md5. For
each fold, Dixon-Coles is fit on the other 4 folds and predicts the held-out
fold; after all 5 fits every game has an out-of-sample prediction.

Firing rule: score = edge = pred(token) − best_ask(token). Fire when
``ASK_LO <= ask <= ASK_HI`` and ``edge > threshold``. Trade universe = all 24
tokens (4 canonical market types — moneyline, spread, totals, btts — both
YES and NO sides). PnL is per unit notional: ``outcome − ask``.

Output: 5 stdout tables (aggregate + one per market type) with columns
``thr, n, totPnL, PnL/sh, SE, t, win%, fold PnL/sh min..max``; plus a
long-format CSV with a ``market_type`` column.

Usage:
    python -m pregame_dc.analyze.edge_threshold_sweep \\
        [--dataset PATH] [--time-seconds T] [--thresholds T1 T2 ...] [--out PATH]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pregame_dc import paths
from pregame_dc.constants import X_COLS, Y_COLS
from pregame_dc.models import dixon_coles as dc

ASK_LO, ASK_HI = 0.01, 0.99
N_FOLDS = 5
LOSS = "brier"

DEFAULT_GRID = np.round(np.arange(0.000, 0.300 + 1e-9, 0.005), 4)

MONEYLINE_SLOTS_Y = [0, 1, 2]
SPREAD_SLOTS_Y = [3, 4, 5, 6]
TOTALS_SLOTS_Y = [7, 8, 9, 10]
BTTS_SLOTS_Y = [11]
ALL_SLOTS_Y = MONEYLINE_SLOTS_Y + SPREAD_SLOTS_Y + TOTALS_SLOTS_Y + BTTS_SLOTS_Y
CANDIDATE_SLOTS = ALL_SLOTS_Y + [s + 12 for s in ALL_SLOTS_Y]
TYPE_SLOTS = {
    "moneyline": MONEYLINE_SLOTS_Y + [s + 12 for s in MONEYLINE_SLOTS_Y],
    "spread":    SPREAD_SLOTS_Y    + [s + 12 for s in SPREAD_SLOTS_Y],
    "totals":    TOTALS_SLOTS_Y    + [s + 12 for s in TOTALS_SLOTS_Y],
    "btts":      BTTS_SLOTS_Y      + [s + 12 for s in BTTS_SLOTS_Y],
}


def fold_of(slug: str) -> int:
    """Deterministic fold assignment (0..N_FOLDS-1) by game_slug md5."""
    return int(hashlib.md5(slug.encode()).hexdigest(), 16) % N_FOLDS


def load_dataset(dataset_path: Path, t_target: float):
    """Return X (n, 24), y (n, 12 YES outcomes), slugs (n,) at t_target.
    Drops rows where the entire 24-vec is sentinel."""
    df = pq.read_table(
        dataset_path,
        columns=["game_slug", "seconds_since_game_start", *X_COLS, *Y_COLS],
    ).to_pandas()
    df = df[df["seconds_since_game_start"] == t_target]
    X = df[X_COLS].to_numpy(dtype=np.float64)
    y = df[Y_COLS].to_numpy(dtype=np.float64)
    keep = ~(X == 1.0).all(axis=1)
    return X[keep], y[keep], df.loc[keep, "game_slug"].to_numpy()


def oos_predictions(X, y, folds):
    """Fit DC on 4 folds, predict the 5th; repeat. Returns pred24 (n, 24)."""
    pred = np.full((len(X), 24), np.nan)
    for f in range(N_FOLDS):
        tr = folds != f
        va = folds == f
        params = dc.fit(X[tr], y[tr], loss=LOSS)
        flag = "" if params["converged"] else "  [WARN: not converged]"
        print(f"  fold {f}: train n={tr.sum():4d}  val n={va.sum():4d}  "
              f"rho={params['rho']:+.4f}  train_{LOSS}={params['train_loss']:.5f}{flag}")
        p12 = dc.predict_probs(X[va], params["mu"], params["sd_safe"],
                               params["w_a"], params["w_b"], params["rho"])
        pred[va] = np.hstack([p12, 1.0 - p12])
    return pred


def collect_cells(X, Y24, pred24, folds):
    """One row per (game, candidate token): (edge, pnl, outcome, fold, slot,
    game_idx). game_idx is the per-game row index used for cluster-robust SE."""
    rows = []
    for s in CANDIDATE_SLOTS:
        ask = X[:, s]
        valid = (ask >= ASK_LO) & (ask <= ASK_HI)
        edge = pred24[:, s] - ask
        pnl = Y24[:, s] - ask
        for i in np.where(valid)[0]:
            rows.append((edge[i], pnl[i], Y24[i, s], folds[i], s, i))
    return np.array(rows, dtype=np.float64)


def _stat(pnl, game_ids):
    """Mean, sum, cluster-robust SE (Liang-Zeger), and t. Clusters by game so
    the SE correctly reflects within-game correlation across cells; raw cell-
    level SE understates this by a meaningful factor."""
    n = len(pnl)
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    mean = pnl.mean()
    if n == 1:
        return n, pnl.sum(), mean, 0.0, 0.0
    residuals = pnl - mean
    # group residuals by game and sum within group
    order = np.argsort(game_ids, kind="stable")
    sorted_ids = game_ids[order]
    sorted_res = residuals[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1, n]
    cluster_sums = np.add.reduceat(sorted_res, boundaries[:-1])
    G = len(cluster_sums)
    if G < 2:
        return n, pnl.sum(), mean, 0.0, 0.0
    var_mean = (G / (G - 1.0)) * (cluster_sums**2).sum() / n**2
    se = float(np.sqrt(max(var_mean, 0.0)))
    t = mean / se if se > 0 else 0.0
    return n, pnl.sum(), mean, se, t


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=paths.TELONEX_LABELED,
                    help="telonex labeled dataset parquet (default: %(default)s)")
    ap.add_argument("--time-seconds", type=float, default=-600.0,
                    help="seconds_since_game_start to evaluate at (default -600)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=None,
                    help="edge threshold grid (default: 0.000..0.300 in 0.005 steps)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output CSV path (default: plots/edge_threshold_sweep_<stem>.csv)")
    args = ap.parse_args()

    grid = (np.round(np.array(args.thresholds), 4) if args.thresholds
            else DEFAULT_GRID)
    out_csv = args.out or (paths.PACKAGE_ROOT / "plots"
                           / f"edge_threshold_sweep_{args.dataset.stem}.csv")

    print(f"loading {args.dataset} at t={args.time_seconds}s ...")
    X, y, slugs = load_dataset(args.dataset, args.time_seconds)
    Y24 = np.hstack([y, 1.0 - y])
    folds = np.array([fold_of(s) for s in slugs])
    print(f"  {len(X)} games  ({len(set(slugs))} unique slugs)")
    print(f"  fold sizes: {[int((folds == f).sum()) for f in range(N_FOLDS)]}")

    print(f"\n{N_FOLDS}-fold CV: fitting Dixon-Coles ({LOSS}) per fold ...")
    pred24 = oos_predictions(X, y, folds)

    cells = collect_cells(X, Y24, pred24, folds)
    edge_arr, pnl, outcome, cell_fold = (cells[:, 0], cells[:, 1],
                                          cells[:, 2], cells[:, 3])
    slot = cells[:, 4].astype(int)
    cell_game = cells[:, 5].astype(np.int64)

    type_masks = {"aggregate": np.ones(len(slot), dtype=bool)}
    for name, slots in TYPE_SLOTS.items():
        type_masks[name] = np.isin(slot, slots)
    print(f"\ncandidate cells (all 4 market types, both sides, ask in "
          f"[{ASK_LO},{ASK_HI}]): {len(cells)}")
    for name, mask in type_masks.items():
        print(f"  {name:>10}: {int(mask.sum())}")

    print("\nThreshold sweep (5-fold OOS, edge rule, score = pred − ask):")
    header = (f"{'thr':>5s} {'n':>6s} {'totPnL':>9s} {'PnL/sh':>9s} "
              f"{'SE':>8s} {'t':>7s} {'win%':>6s} "
              f"{'fold PnL/sh min..max':>22s}")

    out_rows = []
    for type_name, type_mask in type_masks.items():
        print(f"\n=== {type_name} ===")
        print(header)
        print("-" * len(header))
        for thr in grid:
            fire = (edge_arr > thr) & type_mask
            n, tot, mean, se, t = _stat(pnl[fire], cell_game[fire])
            if n == 0:
                print(f"{thr:>5.3f} {0:>6d}  (no fires)")
                out_rows.append([f"{thr:.3f}", type_name, 0,
                                 "", "", "", "", "", "", ""])
                continue
            win = (outcome[fire] == 1).mean() * 100
            fold_means = []
            for f in range(N_FOLDS):
                fm = fire & (cell_fold == f)
                fold_means.append(pnl[fm].mean() if fm.sum() else np.nan)
            fmin = float(np.nanmin(fold_means))
            fmax = float(np.nanmax(fold_means))
            print(f"{thr:>5.3f} {n:>6d} {tot:>+9.2f} {mean:>+9.4f} {se:>8.4f} "
                  f"{t:>+7.2f} {win:>6.1f} "
                  f"{fmin:>+10.4f}..{fmax:>+9.4f}")
            out_rows.append([f"{thr:.3f}", type_name, n,
                             f"{tot:.4f}", f"{mean:.6f}",
                             f"{se:.6f}", f"{t:.4f}", f"{win:.2f}",
                             f"{fmin:.6f}", f"{fmax:.6f}"])

    # Per-type optimum: threshold that maximizes total PnL, but with a
    # noise-robust criterion. For each interior grid point e, take
    # min(totPnL(e-de), totPnL(e), totPnL(e+de)); pick the e that maximizes
    # that. A spurious one-cell PnL spike won't win unless its two neighbors
    # also do well. Requires n >= 30 fires at e *and* both neighbors.
    n_games_total = len(X)
    print(f"\n=== per-type threshold maximizing total PnL "
          f"(robust: max over e of min(tot(e±de), tot(e)); "
          f"n_games={n_games_total}, n>=30 at e and both neighbors) ===")
    summary_header = (f"{'type':<11} {'thr*':>6s} {'n':>5s} "
                      f"{'fire/game':>10s} {'totPnL':>9s} {'PnL/sh':>9s} "
                      f"{'SE':>7s} {'t':>6s} {'win%':>6s}")
    print(summary_header)
    print("-" * len(summary_header))
    for type_name, type_mask in type_masks.items():
        # Precompute per-threshold stats for this type so we can look at
        # neighbors cheaply.
        per_thr = []
        for thr in grid:
            fire = (edge_arr > thr) & type_mask
            n = int(fire.sum())
            if n < 30:
                per_thr.append(None)
                continue
            n_, tot, mean, se, t = _stat(pnl[fire], cell_game[fire])
            win = (outcome[fire] == 1).mean() * 100
            per_thr.append(dict(thr=float(thr), n=n, tot=tot, mean=mean,
                                se=se, t=t, win=win,
                                rate=n / n_games_total))
        best_opt = None
        best_triplet_min = -np.inf
        for i in range(1, len(grid) - 1):
            a, b, c = per_thr[i - 1], per_thr[i], per_thr[i + 1]
            if a is None or b is None or c is None:
                continue
            triplet_min = min(a["tot"], b["tot"], c["tot"])
            if triplet_min > best_triplet_min:
                best_triplet_min = triplet_min
                best_opt = b
        if best_opt is None:
            print(f"{type_name:<11}  (no interior threshold with >=30 fires "
                  f"at all three of e-de, e, e+de)")
            continue
        print(f"{type_name:<11} {best_opt['thr']:>6.3f} {best_opt['n']:>5d} "
              f"{best_opt['rate']:>10.4f} "
              f"{best_opt['tot']:>+9.2f} {best_opt['mean']:>+9.4f} "
              f"{best_opt['se']:>7.4f} {best_opt['t']:>+6.2f} "
              f"{best_opt['win']:>6.1f}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["edge_threshold", "market_type", "n_fires",
                    "total_pnl_per_share", "mean_pnl_per_share", "se",
                    "t_stat", "win_pct",
                    "fold_pnl_per_share_min", "fold_pnl_per_share_max"])
        w.writerows(out_rows)
    print(f"\nwrote {out_csv}")

    agg = [r for r in out_rows
           if r[1] == "aggregate" and r[2] and int(r[2]) >= 30]
    best = max(agg, key=lambda r: float(r[4]), default=None) if agg else None
    if best:
        print(f"\nbest aggregate PnL/share (>=30 fires): threshold {best[0]}  "
              f"n={best[2]}  PnL/share={float(best[4]):+.4f}  t={float(best[6]):+.2f}")
    print("\nPnL/share is per unit notional (outcome − ask). SE/t are "
          "cluster-robust by\ngame_slug (Liang-Zeger), accounting for "
          "within-game cell correlation. The\nper-fold min..max spread is "
          "an additional honest stability check.")


if __name__ == "__main__":
    main()
