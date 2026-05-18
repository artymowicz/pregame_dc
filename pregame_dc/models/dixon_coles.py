"""Dixon-Coles two-rate goal model for the 12 pregame soccer markets.

Each game has two latent rates a>0, b>0 -- the expected goals of team A and
team B.  Goals are Poisson, n_A ~ Pois(a), n_B ~ Pois(b), and the joint pmf
gets the Dixon-Coles low-score correction tau(rho), which re-weights the four
0/1-goal cells to capture the well-known draw / low-score excess:

    tau(0,0) = 1 - a*b*rho    tau(0,1) = 1 + a*rho
    tau(1,0) = 1 + b*rho      tau(1,1) = 1 - rho      (tau = 1 elsewhere)

Every one of the 12 markets is a fixed 0/1 region of the (n_A, n_B) goal
lattice, so its probability is the inner product  p_m = <M, C_m>  of the
joint pmf M with the region indicator C_m.

The rates are linear in the 24 z-scored best asks through a softplus link:

    z = (asks - mu) / sd_safe ;   F = [z | 1]
    a = softplus(F . w_a) ;       b = softplus(F . w_b)

so a fitted model is (mu, sd_safe, w_a, w_b, rho) -- 51 numbers.

This module owns both ends:
  - fit(X, y)            analytic-gradient L-BFGS training  -> live/fit_save_dc.py
  - predict_probs(...)   vectorised inference               -> live/bot.py
  - DixonColesModel      thin .npz loader used by the live bot

The fit gradient is hand-derived (no autodiff dependency) and finite-diff
checked in scratch/poisson_extensions.py, where this model was prototyped.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

NMAX = 25                                    # goal-count truncation (P(>25)~0)
_EPS = 1e-6
_K = np.arange(NMAX + 1)
_LOGFACT = gammaln(_K + 1.0)                 # log k!  for k = 0..NMAX
MEAN_GOALS_INIT = 1.4                        # softplus-intercept warm start


def _region_matrices() -> np.ndarray:
    """C[m, i, j] = 1 iff goal pair (n_A=i, n_B=j) is in market m's region.

    Slot order matches pregame_dc.constants.MARKET_LABELS:
      0 A win  1 B win  2 Draw  3 A-1.5  4 B-1.5  5 A-2.5  6 B-2.5
      7 O1.5   8 O2.5   9 O3.5  10 O4.5   11 BTTS
    """
    i = np.arange(NMAX + 1)[:, None]         # A goals down rows
    j = np.arange(NMAX + 1)[None, :]         # B goals across cols
    d, s = i - j, i + j
    regions = [
        d > 0, d < 0, d == 0,                # A win, B win, Draw
        d >= 2, d <= -2, d >= 3, d <= -3,    # A-1.5, B-1.5, A-2.5, B-2.5
        s >= 2, s >= 3, s >= 4, s >= 5,      # O1.5 .. O4.5
        (i >= 1) & (j >= 1),                 # BTTS
    ]
    return np.stack([r.astype(np.float64) for r in regions])


C = _region_matrices()                        # (12, NMAX+1, NMAX+1)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def _poisson_pmf(lam: np.ndarray) -> np.ndarray:
    """lam (n,) > 0  ->  pmf (n, NMAX+1).  exp(-lam) lam^k / k! via logs."""
    lam = np.asarray(lam, dtype=np.float64)
    log_pmf = -lam[:, None] + _K[None, :] * np.log(lam)[:, None] - _LOGFACT[None, :]
    return np.exp(log_pmf)


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _tau(a: np.ndarray, b: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles correction matrix T (n, NMAX+1, NMAX+1): 1 except 4 cells."""
    n = len(a)
    T = np.ones((n, NMAX + 1, NMAX + 1))
    T[:, 0, 0] = 1.0 - a * b * rho
    T[:, 0, 1] = 1.0 + a * rho
    T[:, 1, 0] = 1.0 + b * rho
    T[:, 1, 1] = 1.0 - rho
    return T


# --------------------------------------------------------------------------
# forward pass
# --------------------------------------------------------------------------
def joint_pmf(a, b, rho: float) -> np.ndarray:
    """rates a,b (n,) and scalar rho  ->  Dixon-Coles joint pmf M (n,N+1,N+1)."""
    a = np.atleast_1d(np.asarray(a, dtype=np.float64))
    b = np.atleast_1d(np.asarray(b, dtype=np.float64))
    pa, pb = _poisson_pmf(a), _poisson_pmf(b)
    return _tau(a, b, rho) * (pa[:, :, None] * pb[:, None, :])


def probs_from_rates(a, b, rho: float) -> np.ndarray:
    """rates -> p (n, 12) YES-side market probabilities."""
    return np.einsum("gij,mij->gm", joint_pmf(a, b, rho), C)


def rates_from_asks(asks, mu, sd_safe, w_a, w_b):
    """asks (24,) or (n,24) raw best-ask vectors -> (a, b), each (n,)."""
    asks = np.atleast_2d(np.asarray(asks, dtype=np.float64))
    z = (asks - mu) / sd_safe
    F = np.hstack([z, np.ones((len(z), 1))])
    return _softplus(F @ w_a), _softplus(F @ w_b)


def predict_probs(asks, mu, sd_safe, w_a, w_b, rho: float) -> np.ndarray:
    """asks (24,) or (n,24)  ->  p (n, 12) YES-side market probabilities."""
    a, b = rates_from_asks(asks, mu, sd_safe, w_a, w_b)
    return probs_from_rates(a, b, rho)


# --------------------------------------------------------------------------
# fitting  (analytic gradient + L-BFGS)
# --------------------------------------------------------------------------
def _loss_grad(theta, F, y, kind, n, d):
    """theta = [w_a (d) | w_b (d) | rho].  Returns (loss, grad (2d+1,))."""
    w_a, w_b, rho = theta[:d], theta[d:2 * d], theta[-1]
    ra, rb = F @ w_a, F @ w_b
    a, b = _softplus(ra), _softplus(rb)
    pa, pb = _poisson_pmf(a), _poisson_pmf(b)
    B = pa[:, :, None] * pb[:, None, :]
    T = _tau(a, b, rho)
    M = T * B                                          # mass-preserving
    p = np.einsum("gij,mij->gm", M, C)

    if kind == "brier":
        loss = np.mean((p - y) ** 2)
        dL_dp = 2.0 * (p - y) / (n * 12)
    elif kind == "ce":
        pc = np.clip(p, _EPS, 1.0 - _EPS)
        loss = -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))
        dL_dp = -(y / pc - (1 - y) / (1 - pc)) / (n * 12)
    else:
        raise ValueError(f"unknown loss {kind!r} (expected 'brier' or 'ce')")

    dM = np.einsum("gm,mij->gij", dL_dp, C)

    # path 1: through the factorised base B
    dL_dB = dM * T
    dL_dpa = np.einsum("gij,gj->gi", dL_dB, pb)
    dL_dpb = np.einsum("gij,gi->gj", dL_dB, pa)
    dpa = np.concatenate([-pa[:, :1], pa[:, :-1] - pa[:, 1:]], axis=1)
    dpb = np.concatenate([-pb[:, :1], pb[:, :-1] - pb[:, 1:]], axis=1)
    dL_da = np.sum(dL_dpa * dpa, axis=1)
    dL_db = np.sum(dL_dpb * dpb, axis=1)

    # path 2: through tau (depends on a, b, rho directly at the 4 corner cells)
    dL_dT = dM * B
    dL_da += dL_dT[:, 0, 0] * (-b * rho) + dL_dT[:, 0, 1] * rho
    dL_db += dL_dT[:, 0, 0] * (-a * rho) + dL_dT[:, 1, 0] * rho
    dL_drho = (dL_dT[:, 0, 0] * (-a * b) + dL_dT[:, 0, 1] * a
               + dL_dT[:, 1, 0] * b + dL_dT[:, 1, 1] * (-1.0))

    grad = np.concatenate([F.T @ (dL_da * _sigmoid(ra)),
                           F.T @ (dL_db * _sigmoid(rb)),
                           [dL_drho.sum()]])
    return loss, grad


def fit(X: np.ndarray, y: np.ndarray, loss: str = "brier") -> dict:
    """Fit Dixon-Coles on raw asks X (n,24) and YES outcomes y (n,12).

    Returns a dict of fitted parameters + provenance, ready for np.savez.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd_safe = np.where(sd > 0, sd, 1.0)
    F = np.hstack([(X - mu) / sd_safe, np.ones((len(X), 1))])
    n, d = len(X), F.shape[1]

    c0 = float(np.log(np.expm1(MEAN_GOALS_INIT)))      # softplus(c0) ~ 1.4
    theta0 = np.zeros(2 * d + 1)
    theta0[d - 1] = c0                                 # w_a intercept
    theta0[2 * d - 1] = c0                             # w_b intercept

    res = minimize(
        lambda th: _loss_grad(th, F, y, loss, n, d),
        theta0, jac=True, method="L-BFGS-B",
        options={"maxiter": 4000, "ftol": 1e-13, "gtol": 1e-10},
    )
    return {
        "mu": mu,
        "sd_safe": sd_safe,
        "w_a": res.x[:d],
        "w_b": res.x[d:2 * d],
        "rho": float(res.x[-1]),
        "loss": loss,
        "converged": bool(res.success),
        "n_iter": int(res.nit),
        "train_loss": float(res.fun),
    }


# --------------------------------------------------------------------------
# live-bot loader
# --------------------------------------------------------------------------
class DixonColesModel:
    """Loads fitted Dixon-Coles parameters from an .npz and predicts.

    `predict(asks_24)` mirrors the contract the live bot expects from a model:
    it takes a raw 24-element ask vector (canonical token order) and returns a
    24-element vector of token probabilities -- the 12 YES probabilities
    followed by the 12 NO probabilities (1 - YES).
    """

    def __init__(self, npz_path: Path):
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Dixon-Coles model file not found: {npz_path}. "
                f"Run `python -m pregame_dc.live.fit_save_dc` first."
            )
        z = np.load(npz_path)
        self.mu = z["mu"]                              # (24,)
        self.sd_safe = z["sd_safe"]                    # (24,)
        self.w_a = z["w_a"]                            # (25,)
        self.w_b = z["w_b"]                            # (25,)
        self.rho = float(z["rho"])
        self.t_target = float(z["T_TARGET"])
        self.train_n = int(z["train_n"])
        self.loss = str(z["loss"]) if "loss" in z.files else "brier"

    def predict(self, asks_24: np.ndarray) -> np.ndarray:
        """asks_24: (24,) raw ask vector -> (24,) token predictions."""
        p12 = predict_probs(
            asks_24, self.mu, self.sd_safe, self.w_a, self.w_b, self.rho
        )[0]
        return np.concatenate([p12, 1.0 - p12])
