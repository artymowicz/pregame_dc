"""Cross-eval the Dixon-Coles model on the best_of feed.

Train on telonex games NOT in self_collected; validate on the best_of feed
over the telonex <-> self_collected overlap.

best_of = telonex, but every sentinel ask (== 1.0, "no book") is replaced by
the self_collected ask for the same (game, token) when self_collected has a
real value there. It is the cleanest eval feed: telonex's game coverage with
its quote gaps patched from the higher-quality self_collected feed. Same
construction as scratch/best_of_dataset.py, rebuilt here in-memory at the
single evaluation timepoint so this script needs only the two shipped labeled
parquets (no scratch artefact).

Because every best_of game is a self_collected game, and the training set
excludes all self_collected slugs, the evaluation is held out by game identity
-- a genuine out-of-sample check.

Companion to cross_eval_self_collected_dc.py, which instead evaluates on the
raw self_collected feed with a paired self_collected-vs-telonex comparison.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pregame_dc import paths
from pregame_dc.constants import X_COLS, Y_COLS, TYPE_FOR_SLOT
from pregame_dc.models import dixon_coles as dc

T_TARGET = -600.0
PLACEHOLDER = 1.0
ASK_LO, ASK_HI = 0.01, 0.99
THRESHOLDS = [0.00, 0.02, 0.05, 0.10, 0.15]
LOSS = "brier"


def _load_at_t(path: Path, with_y: bool):
    cols = ["game_slug", "seconds_since_game_start", *X_COLS]
    if with_y:
        cols += Y_COLS
    df = pq.read_table(path, columns=cols).to_pandas()
    df = df[df["seconds_since_game_start"] == T_TARGET].drop_duplicates("game_slug")
    return df.set_index("game_slug")


def load_train(exclude_slugs):
    """telonex at T_TARGET minus `exclude_slugs`; -> X (n,24), y (n,12)."""
    df = _load_at_t(paths.TELONEX_LABELED, with_y=True)
    df = df[~df.index.isin(exclude_slugs)]
    X = df[X_COLS].to_numpy(dtype=np.float64)
    y = df[Y_COLS].to_numpy(dtype=np.float64)
    keep = ~(X == PLACEHOLDER).all(axis=1)
    return X[keep], y[keep]


def build_best_of():
    """best_of feed at T_TARGET over the telonex/self_collected overlap.

    Returns X (m,24) asks, y (m,12) outcomes, and a small stats dict.
    """
    tx = _load_at_t(paths.TELONEX_LABELED, with_y=True)
    sc = _load_at_t(paths.SELF_COLLECTED_LABELED, with_y=False)
    overlap = sorted(set(tx.index) & set(sc.index))
    tx, sc = tx.loc[overlap], sc.loc[overlap]

    X_tx = tx[X_COLS].to_numpy(dtype=np.float64)
    X_sc = sc[X_COLS].to_numpy(dtype=np.float64)
    y = tx[Y_COLS].to_numpy(dtype=np.float64)

    # Substitute where telonex is a sentinel AND self_collected has a real value.
    sub = (X_tx == PLACEHOLDER) & np.isfinite(X_sc) & (X_sc != PLACEHOLDER)
    X_bo = np.where(sub, X_sc, X_tx)

    keep = ~(X_bo == PLACEHOLDER).all(axis=1)
    n_tx_sent = int((X_tx == PLACEHOLDER).sum())
    stats = {
        "overlap_games": len(overlap),
        "kept_games": int(keep.sum()),
        "tx_sentinels": n_tx_sent,
        "repaired_from_sc": int(sub.sum()),
        "still_sentinel": int((X_bo == PLACEHOLDER).sum()),
    }
    return X_bo[keep], y[keep], stats


def fit_dc(X_tr, y_tr):
    """Fit Dixon-Coles; return a predict closure X_eval (n,24) -> (n,24)."""
    params = dc.fit(X_tr, y_tr, loss=LOSS)
    if not params["converged"]:
        print(f"[warn] DC fit did not converge ({params['n_iter']} iters)")

    def predict(X_eval):
        p12 = dc.predict_probs(
            X_eval, params["mu"], params["sd_safe"],
            params["w_a"], params["w_b"], params["rho"],
        )
        return np.hstack([p12, 1.0 - p12])
    return predict, params


def report(Y_tr, X_va, Y_va, predict):
    """Brier over the 12 YES markets + an edge-rule PnL-by-threshold table."""
    pred_va = predict(X_va)
    yhat = np.clip(pred_va, 0.0, 1.0)

    yes_idx = list(range(12))
    base_yhat = np.tile(Y_tr.mean(axis=0), (len(Y_va), 1))
    base_mse = ((base_yhat - Y_va) ** 2).mean(axis=0)
    mse = ((yhat - Y_va) ** 2).mean(axis=0)
    ss_tot = ((Y_va - Y_va.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ((yhat - Y_va) ** 2).sum(axis=0) / np.where(ss_tot > 0, ss_tot, 1.0)

    print(f"Mean Brier (12 YES): base={base_mse[yes_idx].mean():.4f}  "
          f"DC={mse[yes_idx].mean():.4f}  R2={r2[yes_idx].mean():+.4f}")

    ask = X_va
    edge = pred_va - ask
    pnl = Y_va - ask
    valid = (ask >= ASK_LO) & (ask <= ASK_HI)

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


def main():
    sc_slugs = set(
        pq.read_table(paths.SELF_COLLECTED_LABELED, columns=["game_slug"])
        .to_pandas()["game_slug"].unique()
    )

    # Train: telonex MINUS self_collected overlap (clean OOS by game identity).
    X_tr, y_tr = load_train(exclude_slugs=sc_slugs)
    predict, params = fit_dc(X_tr, y_tr)
    print(f"Dixon-Coles fit ({LOSS} loss): train n={len(X_tr)}  "
          f"rho={params['rho']:+.4f}  train_{LOSS}={params['train_loss']:.5f}")

    # Validate: best_of feed over the overlap.
    X_bo, y_bo, st = build_best_of()
    print(f"best_of eval: {st['kept_games']} games "
          f"(of {st['overlap_games']} overlap)  "
          f"telonex sentinels={st['tx_sentinels']}, "
          f"repaired from self_collected={st['repaired_from_sc']}, "
          f"still sentinel={st['still_sentinel']}")

    Y_tr = np.hstack([y_tr, 1.0 - y_tr])
    Y_bo = np.hstack([y_bo, 1.0 - y_bo])
    print(f"\n{'='*60}\neval = best_of  (train telonex - self_collected)\n{'='*60}")
    print(f"train n={len(Y_tr)}   val n={len(Y_bo)}")
    report(Y_tr, X_bo, Y_bo, predict)


if __name__ == "__main__":
    main()
