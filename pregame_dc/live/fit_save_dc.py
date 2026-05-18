"""Train the Dixon-Coles goal model on telonex and save its parameters.

The live bot loads `mu`, `sd_safe`, `w_a`, `w_b`, `rho` from the resulting
.npz (see pregame_dc.models.dixon_coles.DixonColesModel) -- so this is run
once offline, not on every bot start.

By default trains on the full telonex set at t = -10 min with Brier loss.
`--exclude-self-collected` drops games that also appear in the self_collected
dataset (useful when self_collected is being held out for evaluation).

Outcomes are read from the labeled parquet's `y_0..y_11` columns directly
(the 12 YES-side market outcomes); Dixon-Coles consumes those 12 columns as-is.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pregame_dc import paths
from pregame_dc.constants import X_COLS, Y_COLS
from pregame_dc.models import dixon_coles as dc


def load_telonex(t_target: float, exclude_self_collected: bool = False):
    """Returns (X, y): X is (n,24) raw asks, y is (n,12) YES outcomes, at the
    given t-offset on the telonex labeled dataset. Fully-placeholder rows
    (all 24 asks == 1.0) are dropped."""
    df = pq.read_table(
        paths.TELONEX_LABELED,
        columns=["game_slug", "seconds_since_game_start", *X_COLS, *Y_COLS],
    ).to_pandas()
    df = df[df["seconds_since_game_start"] == t_target]
    if exclude_self_collected:
        sc_slugs = set(
            pq.read_table(paths.SELF_COLLECTED_LABELED, columns=["game_slug"])
            .to_pandas()["game_slug"].unique()
        )
        df = df[~df["game_slug"].isin(sc_slugs)]
    X = df[X_COLS].to_numpy(dtype=np.float64)
    y = df[Y_COLS].to_numpy(dtype=np.float64)
    keep = ~(X == 1.0).all(axis=1)
    return X[keep], y[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-seconds", type=float, default=-600.0,
                    help="seconds_since_game_start to train at (default -600 = -10min)")
    ap.add_argument("--out", type=Path, default=paths.MODEL_DC_T_10MIN)
    ap.add_argument("--loss", choices=["brier", "ce"], default="brier",
                    help="training loss (default brier)")
    ap.add_argument("--exclude-self-collected", action="store_true",
                    help="Drop games that also appear in self_collected from "
                         "the training set (off by default).")
    args = ap.parse_args()

    msg = f"loading telonex at t={args.time_seconds}s"
    if args.exclude_self_collected:
        msg += " (excluding self_collected overlap)"
    print(msg + " ...")
    X_tr, y_tr = load_telonex(args.time_seconds, args.exclude_self_collected)
    print(f"  train n = {len(X_tr)} games")

    print(f"fitting Dixon-Coles ({args.loss} loss) ...")
    p = dc.fit(X_tr, y_tr, loss=args.loss)
    print(f"  {'converged' if p['converged'] else 'DID NOT CONVERGE'} "
          f"in {p['n_iter']} iterations,  train {args.loss}={p['train_loss']:.5f}")
    print(f"  rho = {p['rho']:+.4f}")
    a, b = dc.rates_from_asks(X_tr, p["mu"], p["sd_safe"], p["w_a"], p["w_b"])
    print(f"  fitted goal rates: a mean={a.mean():.2f}  b mean={b.mean():.2f}  "
          f"total mean={(a + b).mean():.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        mu=p["mu"],
        sd_safe=p["sd_safe"],
        w_a=p["w_a"],
        w_b=p["w_b"],
        rho=np.array(p["rho"]),
        loss=np.array(args.loss),
        T_TARGET=np.array(args.time_seconds),
        train_n=np.array(len(X_tr)),
    )
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
