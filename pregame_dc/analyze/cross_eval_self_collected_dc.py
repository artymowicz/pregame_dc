"""Clean cross-eval of the Dixon-Coles model: train on telonex games NOT in
self_collected, eval on self_collected.

Dixon-Coles port of cross_eval_self_collected_clean.py (which evaluates the
legacy rank-3 PCR). Everything except the model is identical, so the two
scripts are directly comparable.

~96.5% of self_collected games overlap with telonex (same Polymarket games,
different data feeds). Training only on the non-overlapping telonex games
makes the self_collected evaluation truly held out by game identity. With
~2200 train games and 51 parameters in Dixon-Coles (w_a 25, w_b 25, rho 1),
overfitting risk is intrinsically low; this script confirms the leakage isn't
driving conclusions.

Output, twice (once scoring on the self_collected feed, once on telonex,
both restricted to the same both-feeds-quote cells so the comparison is
apples-to-apples):
  - Mean Brier over the 12 YES markets: base rate vs Dixon-Coles, plus R2.
  - PnL-by-threshold table under the edge rule (buy at ask when
    pred - ask > threshold; pnl/trade = outcome - ask), split by market type.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pregame_dc import paths
from pregame_dc.constants import X_COLS, Y_COLS, TYPE_FOR_SLOT
from pregame_dc.models import dixon_coles as dc

T_TARGET = -600.0
ASK_LO, ASK_HI = 0.01, 0.99
THRESHOLDS = [0.00, 0.02, 0.05, 0.10, 0.15]
LOSS = "brier"


def load(path: Path, exclude_slugs=None):
    df = pq.read_table(
        path,
        columns=["game_slug", "split", "seconds_since_game_start", *X_COLS, *Y_COLS],
    ).to_pandas()
    df = df[df["seconds_since_game_start"] == T_TARGET]
    if exclude_slugs is not None:
        df = df[~df["game_slug"].isin(exclude_slugs)]
    X = df[X_COLS].to_numpy(dtype=np.float64)
    y = df[Y_COLS].to_numpy(dtype=np.float64)
    keep = ~(X == 1.0).all(axis=1)
    Y = np.hstack([y[keep], 1.0 - y[keep]])
    return X[keep], Y, df.loc[keep, "game_slug"].to_numpy()


def fit_dc(X_tr, Y_tr):
    """Fit Dixon-Coles on the training games. `Y_tr` is the 24-token outcome
    matrix; DC consumes the first 12 columns (the YES outcomes). Returns a
    `predict` closure that maps X_eval (n,24) -> (n,24) token predictions,
    matching the PCR `fit_topk` contract."""
    params = dc.fit(X_tr, Y_tr[:, :12], loss=LOSS)
    if not params["converged"]:
        print(f"[warn] DC fit did not converge ({params['n_iter']} iters)")

    def predict(X_eval):
        p12 = dc.predict_probs(
            X_eval, params["mu"], params["sd_safe"],
            params["w_a"], params["w_b"], params["rho"],
        )
        return np.hstack([p12, 1.0 - p12])
    return predict, params


def report_paired(name, X_tr, Y_tr, X_va, Y_va, both_quote, predict):
    """Eval restricted to cells where both sources have a valid quote.

    `both_quote` is the per-cell mask common to SC and TX. The fire mask is
    `both_quote & (ask in [ASK_LO, ASK_HI]) & (edge > thr)`. Same cells
    for SC and TX runs, so the only thing that varies between paired
    reports is `X_va` (asks + model input), which drives `pred_va`, edge,
    and the ask paid for fill.
    """
    pred_va = predict(X_va)
    yhat = np.clip(pred_va, 0.0, 1.0)

    yes_idx = list(range(12))
    base_yhat = np.tile(Y_tr.mean(axis=0), (len(Y_va), 1))
    base_mse = ((base_yhat - Y_va) ** 2).mean(axis=0)
    mse = ((yhat - Y_va) ** 2).mean(axis=0)
    ss_tot = ((Y_va - Y_va.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ((yhat - Y_va) ** 2).sum(axis=0) / np.where(ss_tot > 0, ss_tot, 1.0)

    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(f"train n={len(Y_tr)}   val n={len(Y_va)}   both-quote cells={int(both_quote.sum())}")
    print(f"Mean Brier (12 YES): base={base_mse[yes_idx].mean():.4f}  "
          f"DC={mse[yes_idx].mean():.4f}  R2={r2[yes_idx].mean():+.4f}")

    ask = X_va
    edge = pred_va - ask
    pnl = Y_va - ask
    valid = both_quote & (ask >= ASK_LO) & (ask <= ASK_HI)

    type_to_slots = {}
    for slot, mt in TYPE_FOR_SLOT.items():
        for s in (slot, slot + 12):
            type_to_slots.setdefault(mt, []).append(s)

    print(f"\n{'thresh':>7s}  {'agg n/$':>16s}  {'mny n/$':>16s}  "
          f"{'spr n/$':>16s}  {'tot n/$':>16s}  {'btts n/$':>16s}")
    for thr in THRESHOLDS:
        fire = valid & (edge > thr)
        cells = []
        n_all = int(fire.sum())
        cells.append(f"{n_all} / {pnl[fire].mean():+.4f}" if n_all else "0 / n/a")
        for mt in ("moneyline", "spread", "totals", "btts"):
            slots = type_to_slots[mt]
            mask = np.zeros_like(fire)
            mask[:, slots] = fire[:, slots]
            n = int(mask.sum())
            cells.append(f"{n} / {pnl[mask].mean():+.4f}" if n else "0 / n/a")
        print(f"{thr:>7.2f}  " + "  ".join(c.rjust(16) for c in cells))


def load_aligned_eval(sc_path, tx_path, overlap_slugs):
    """Load both eval sources at T_TARGET, inner-joined on game_slug so rows
    are aligned. Drops games where either source is fully placeholder
    (all 24 asks == 1.0). Y is the same outcome vector for both."""
    def _load(path):
        df = pq.read_table(
            path,
            columns=["game_slug", "seconds_since_game_start", *X_COLS, *Y_COLS],
        ).to_pandas()
        df = df[(df["seconds_since_game_start"] == T_TARGET)
                & (df["game_slug"].isin(overlap_slugs))]
        X = df[X_COLS].to_numpy(dtype=np.float64)
        keep = ~(X == 1.0).all(axis=1)
        return df.loc[keep, ["game_slug", *X_COLS, *Y_COLS]].reset_index(drop=True)

    sc = _load(sc_path)
    tx = _load(tx_path)
    merged = sc.merge(tx, on="game_slug", suffixes=("_sc", "_tx"))

    X_sc = merged[[f"{c}_sc" for c in X_COLS]].to_numpy(dtype=np.float64)
    X_tx = merged[[f"{c}_tx" for c in X_COLS]].to_numpy(dtype=np.float64)
    y_sc = merged[[f"{c}_sc" for c in Y_COLS]].to_numpy(dtype=np.float64)
    y_tx = merged[[f"{c}_tx" for c in Y_COLS]].to_numpy(dtype=np.float64)
    # Outcome should be identical between sources for the same game; verify.
    assert np.array_equal(y_sc, y_tx), "outcome mismatch between SC and TX on overlap"
    Y = np.hstack([y_sc, 1.0 - y_sc])
    both_quote = (X_sc < 1.0) & (X_tx < 1.0)
    return X_sc, X_tx, Y, both_quote, merged["game_slug"].to_numpy()


def main():
    self_collected_slugs = set(
        pq.read_table(paths.SELF_COLLECTED_LABELED, columns=["game_slug"])
        .to_pandas()["game_slug"].unique()
    )
    telonex_slugs = set(
        pq.read_table(paths.TELONEX_LABELED, columns=["game_slug"])
        .to_pandas()["game_slug"].unique()
    )
    overlap = self_collected_slugs & telonex_slugs

    # Train: telonex MINUS self_collected overlap (clean OOS).
    X_tr, Y_tr, _ = load(paths.TELONEX_LABELED, exclude_slugs=self_collected_slugs)
    predict, params = fit_dc(X_tr, Y_tr)
    print(f"Dixon-Coles fit ({LOSS} loss): train n={len(X_tr)}  "
          f"rho={params['rho']:+.4f}  train_{LOSS}={params['train_loss']:.5f}")

    X_sc, X_tx, Y, both_quote, slugs = load_aligned_eval(
        paths.SELF_COLLECTED_LABELED, paths.TELONEX_LABELED, overlap,
    )
    n_cells = int(both_quote.sum())
    n_total = both_quote.size
    print(f"aligned eval: {len(slugs)} games, "
          f"{n_cells:,} BOTH-quote cells of {n_total:,} ({n_cells/n_total*100:.1f}%)")

    report_paired(
        "eval = SELF_COLLECTED, restricted to BOTH-quote cells (apples-to-apples vs TX)",
        X_tr, Y_tr, X_sc, Y, both_quote, predict,
    )
    report_paired(
        "eval = TELONEX, restricted to BOTH-quote cells (apples-to-apples vs SC)",
        X_tr, Y_tr, X_tx, Y, both_quote, predict,
    )


if __name__ == "__main__":
    main()
