from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class SIMIWeights:
    spillover: float = 0.5
    centrality: float = 0.5


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def _fit_weights_ridge_simplex(x: np.ndarray, y: np.ndarray, n: int, lam: float, w0: np.ndarray) -> np.ndarray:
    """
    w >= 0, sum(w) = 1, minimize (fraction of variance unexplained) + lam * ||w - uniform||^2.
    Scaling the fit term by var(y)*n_obs puts it on the same order of
    magnitude as the ridge penalty (which maxes out around 1), so a given
    `lam` has a consistent, interpretable meaning across series/subsamples
    instead of being swamped by an ~1e6-scale raw sum-of-squares term.
    """
    uniform = np.full(n, 1.0 / n)
    y_var_n = float(np.var(y) * len(y)) or 1.0

    def objective(w: np.ndarray) -> float:
        resid = x @ w - y
        return float((resid @ resid) / y_var_n + lam * np.sum((w - uniform) ** 2))

    def grad(w: np.ndarray) -> np.ndarray:
        resid = x @ w - y
        return 2.0 * (x.T @ resid) / y_var_n + 2.0 * lam * (w - uniform)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        res = minimize(objective, w0, jac=grad, bounds=bounds, constraints=constraints, method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    out = res.x if (res.success and np.all(np.isfinite(res.x))) else w0
    return out


def estimate_cipi_basket_shares(
    panel: pd.DataFrame,
    material_columns: list[str],
    target_col: str = "CIPI_MAT",
) -> pd.Series:
    """
    Recover an implied CIPI basket-share vector w (w_i >= 0, sum w_i = 1) such
    that CIPI_MAT_t ~= sum_i w_i * material_i_t, using ridge-regularized,
    simplex-constrained least squares.

    UBOS does not publish the official per-material BOQ weights, so this is a
    proxy, not ground truth. An earlier unregularized NNLS version of this
    function was numerically broken two ways: (1) it standardized predictors
    to mean=1 but left the target on its raw ~100-130 index scale, which made
    the (0, 1) box constraint infeasible and drove every weight to its upper
    bound (silently degenerating to equal weights after the final
    normalization); (2) even after fixing that scale mismatch, unregularized
    NNLS on ~18 highly collinear, co-trending CIPI sub-indices is an
    ill-posed inverse problem that collapses onto whichever single series
    happens to trace CIPI_MAT most closely (weight ~0.96 on one material,
    ~0 on the rest) -- a multicollinearity artifact, not a economically
    plausible basket. The ridge-toward-uniform penalty here shrinks the
    solution back toward equal weights in directions the data cannot
    identify, with the ridge strength chosen by time-series holdout
    validation rather than fixed arbitrarily. Treat the output as a weak,
    data-driven proxy for economic significance, not a precise weight.
    """
    if len(material_columns) == 0:
        return pd.Series(dtype=float)

    n = len(material_columns)
    uniform = pd.Series(1.0 / n, index=material_columns)

    if target_col not in panel.columns:
        return uniform

    x = panel[material_columns].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(panel[target_col], errors="coerce")
    keep = x.notna().all(axis=1) & y.notna()
    x = x.loc[keep]
    y = y.loc[keep]

    if len(x) < 20:
        return uniform

    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    w0 = np.full(n, 1.0 / n)

    # Time-series holdout (last 20% of months) to pick the ridge strength
    # from a fixed grid, rather than fitting one lambda in-sample.
    split = max(int(len(x_arr) * 0.8), n + 1)
    if split >= len(x_arr):
        split = len(x_arr) - 5
    x_train, x_val = x_arr[:split], x_arr[split:]
    y_train, y_val = y_arr[:split], y_arr[split:]

    lambdas = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    best_lam, best_mse = lambdas[-1], np.inf
    if len(x_val) >= 3:
        for lam in lambdas:
            w = _fit_weights_ridge_simplex(x_train, y_train, n, lam, w0)
            mse = float(np.mean((x_val @ w - y_val) ** 2))
            if mse < best_mse:
                best_mse, best_lam = mse, lam

    w_final = _fit_weights_ridge_simplex(x_arr, y_arr, n, best_lam, w0)
    if (not np.all(np.isfinite(w_final))) or w_final.sum() <= 0:
        return uniform

    w_final = np.clip(w_final, 0.0, None)
    w_final = w_final / w_final.sum()
    return pd.Series(w_final, index=material_columns)


def compute_simi(
    spill_transmission: pd.Series,
    centrality_composite: pd.Series,
    basket_shares: pd.Series,
    weights: SIMIWeights,
) -> pd.DataFrame:
    idx = sorted(set(spill_transmission.index) | set(centrality_composite.index) | set(basket_shares.index))
    spill = spill_transmission.reindex(idx).fillna(0.0)
    cent = centrality_composite.reindex(idx).fillna(0.0)
    bshare = basket_shares.reindex(idx).fillna(0.0)

    spill_z = zscore(spill)
    cent_z = zscore(cent)

    core = weights.spillover * spill_z + weights.centrality * cent_z
    simi = core * (1.0 + bshare)

    out = pd.DataFrame(
        {
            "spillover_transmission": spill,
            "spillover_z": spill_z,
            "centrality": cent,
            "centrality_z": cent_z,
            "basket_share": bshare,
            "simi_score": simi,
        }
    ).sort_values("simi_score", ascending=False)

    out["rank"] = range(1, len(out) + 1)
    return out


def weighting_scenarios() -> Mapping[str, SIMIWeights]:
    return {
        "equal": SIMIWeights(0.5, 0.5),
        "spillover_heavy": SIMIWeights(0.7, 0.3),
        "centrality_heavy": SIMIWeights(0.3, 0.7),
    }
