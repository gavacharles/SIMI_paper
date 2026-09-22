from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.api import VAR


AGGREGATE_COLUMNS = {
    "CIPI_ALL",
    "CIPI_MAT",
    "CIPI_BLDG",
    "CIPI_CIVIL",
    "CIPI_RES",
    "CIPI_NONRES",
}


@dataclass
class SpilloverConfig:
    lags: int = 2
    horizon: int = 10
    min_obs: int = 48
    # "ols": classical unrestricted VAR (statsmodels), appropriate only when
    #   T >> k^2 * lags. "lasso": elastic-net-regularized VAR per equation
    #   (Demirer, Diebold, Liu & Yilmaz 2018 style), appropriate for the
    #   high-dimensional/short-T regime this panel is actually in.
    #   "auto" switches to lasso once k * lags exceeds ~40% of T, which is
    #   where the unrestricted OLS system starts becoming rank-deficient.
    method: Literal["auto", "ols", "lasso"] = "auto"
    lasso_cv_splits: int = 5


def identify_material_columns(columns: Sequence[str]) -> list[str]:
    cols = []
    for col in columns:
        if not isinstance(col, str):
            continue
        if col in AGGREGATE_COLUMNS:
            continue
        if col.isupper() and col.isalpha() and len(col) >= 3:
            cols.append(col)
    return sorted(cols)


def log_returns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """
    Log-difference for strictly positive price-level columns (material
    sub-indices, exchange rate, commodity benchmarks). Columns containing
    non-positive values are assumed to already be a signed, roughly-
    stationary index (e.g. GSCPI, a standardized z-score that oscillates
    around 0) rather than a price level -- log() of a non-positive value is
    undefined, and masking negatives to NaN (the previous behavior) silently
    drops such a column from the whole system once its NaN share exceeds
    the coverage filter, without any warning that it happened. Those columns
    get a plain first difference instead, which is the correct transform for
    an already-stationary index and keeps the column in the estimation.
    """
    x = df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    has_nonpositive = (x <= 0).any(axis=0)

    price_cols = x.columns[~has_nonpositive]
    signed_cols = x.columns[has_nonpositive]

    parts = []
    if len(price_cols) > 0:
        parts.append(np.log(x[price_cols]).diff())
    if len(signed_cols) > 0:
        parts.append(x[signed_cols].diff())

    out = pd.concat(parts, axis=1)[list(columns)] if parts else x.iloc[0:0]
    return out.dropna(how="all")


def _ma_matrices(var_res, horizon: int) -> np.ndarray:
    k = var_res.neqs
    p = var_res.k_ar
    a = np.nan_to_num(var_res.coefs, nan=0.0, posinf=0.0, neginf=0.0)
    a = np.clip(a, -5.0, 5.0)

    phi = np.zeros((horizon, k, k))
    phi[0] = np.eye(k)

    for h in range(1, horizon):
        acc = np.zeros((k, k))
        for lag in range(1, min(p, h) + 1):
            left = np.clip(phi[h - lag], -10.0, 10.0)
            right = np.clip(a[lag - 1], -10.0, 10.0)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                acc += left @ right
        phi[h] = np.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)
        phi[h] = np.clip(phi[h], -50.0, 50.0)

    return phi


def generalized_fevd(var_res, horizon: int = 10) -> pd.DataFrame:
    """
    Pesaran-Shin generalized FEVD used in Diebold-Yilmaz spillovers.
    Returns a row-normalized matrix in percentages.
    """
    sigma = np.asarray(var_res.sigma_u)
    names = list(var_res.names)
    k = sigma.shape[0]
    phi = _ma_matrices(var_res, horizon)

    theta = np.zeros((k, k))
    for i in range(k):
        e_i = np.zeros((k, 1))
        e_i[i, 0] = 1.0

        den = 0.0
        for h in range(horizon):
            den += (e_i.T @ phi[h] @ sigma @ phi[h].T @ e_i).item()

        for j in range(k):
            e_j = np.zeros((k, 1))
            e_j[j, 0] = 1.0
            num = 0.0
            for h in range(horizon):
                term = (e_i.T @ phi[h] @ sigma @ e_j).item()
                num += term**2
            theta[i, j] = (num / sigma[j, j]) / den if den > 0 else 0.0

    row_sums = theta.sum(axis=1, keepdims=True)
    theta = np.divide(theta, row_sums, out=np.zeros_like(theta), where=row_sums > 0)
    return pd.DataFrame(theta * 100.0, index=names, columns=names)


def _lag_design(data: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (Y, X) for Y_t = [X_t] beta, X_t = concat(y_{t-1}, ..., y_{t-lags})."""
    n, k = data.shape
    y = data[lags:]
    x = np.concatenate([data[lags - lag : n - lag] for lag in range(1, lags + 1)], axis=1)
    return y, x


def fit_lasso_var(clean: pd.DataFrame, lags: int, cv_splits: int = 5) -> SimpleNamespace:
    """
    Elastic-net-style regularized VAR, fit one equation (target variable) at a
    time via cross-validated Lasso on the stacked lag design. This is the
    standard fix for k^2*lags >> T instability in high-dimensional spillover
    networks (Demirer, Diebold, Liu & Yilmaz, 2018, J. Econometrics).

    Returns an object exposing the same attributes fit_spillover_table's
    downstream code (generalized_fevd) needs from a statsmodels VARResults:
    .coefs (lags, k, k), .sigma_u (k, k), .names, .neqs, .k_ar.
    """
    names = list(clean.columns)
    k = len(names)
    y_full, x_full = _lag_design(clean.to_numpy(dtype=float), lags)
    n_obs = y_full.shape[0]

    n_splits = max(2, min(cv_splits, n_obs - 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Lagged regressors span very different native scales/volatilities (e.g.
    # cement returns vs. exchange-rate returns), which makes unscaled
    # coordinate descent numerically unstable (observed as inf/nan in the
    # predict step). Scale each predictor column by its std before fitting
    # and undo the scaling on the recovered coefficients.
    x_std = x_full.std(axis=0)
    x_std_safe = np.where(x_std < 1e-10, 1.0, x_std)
    x_scaled = x_full / x_std_safe

    coefs_by_eq = np.zeros((k, k * lags))
    resid = np.zeros((n_obs, k))
    for j in range(k):
        target = y_full[:, j]
        try:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                model = LassoCV(cv=tscv, max_iter=100000, tol=1e-5, n_alphas=50, random_state=0)
                model.fit(x_scaled, target)
                beta = model.coef_ / x_std_safe
                if not np.all(np.isfinite(beta)):
                    raise ValueError("non-finite Lasso coefficients")
                fitted = x_full @ beta + model.intercept_
            if not np.all(np.isfinite(fitted)):
                raise ValueError("non-finite fitted values")
        except Exception:
            beta = np.zeros(k * lags)
            fitted = np.zeros(n_obs)
        coefs_by_eq[j, :] = beta
        resid[:, j] = target - fitted

    # coefs_by_eq columns are ordered [lag1_var1..lag1_vark, lag2_var1.., ...];
    # reshape into statsmodels' coefs[lag-1][eq, var] convention.
    coefs = np.zeros((lags, k, k))
    for lag in range(lags):
        coefs[lag] = coefs_by_eq[:, lag * k : (lag + 1) * k]

    sigma_u = np.cov(resid, rowvar=False)
    if sigma_u.ndim == 0:
        sigma_u = np.array([[float(sigma_u)]])

    return SimpleNamespace(neqs=k, k_ar=lags, coefs=coefs, sigma_u=sigma_u, names=names)


def prepare_returns(returns: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    """
    Shared cleaning step (coverage filter, short-gap fill, winsorize) so that
    spillover (VAR/FEVD) and correlation/centrality estimation operate over
    the *same* node set and the *same* rows. Previously the correlation step
    ran plain `.dropna()` over the raw returns independently, which for a
    ~20-column panel with staggered coverage could leave fewer valid rows
    than columns -- a guaranteed-singular sample covariance matrix.
    """
    return _prepare_returns(returns, min_obs)


def _prepare_returns(returns: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    clean = returns.copy()
    keep = [c for c in clean.columns if clean[c].notna().sum() >= min_obs]
    clean = clean[keep]
    clean = clean.ffill(limit=2).bfill(limit=2).dropna()
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()

    # Remove nearly constant columns that can create singular/unstable VAR fits.
    keep_var = clean.std(axis=0) > 1e-8
    clean = clean.loc[:, keep_var]

    # Winsorize to reduce extreme splice outliers that destabilize VAR dynamics.
    for col in clean.columns:
        lo, hi = clean[col].quantile([0.01, 0.99])
        clean[col] = clean[col].clip(lower=lo, upper=hi)

    return clean


def fit_spillover_table(returns: pd.DataFrame, cfg: SpilloverConfig) -> pd.DataFrame:
    clean = _prepare_returns(returns, cfg.min_obs)
    if clean.shape[1] < 2:
        raise ValueError("Need at least two valid series for spillover estimation.")

    n_obs, k = clean.shape
    method = cfg.method
    if method == "auto":
        # Unrestricted VAR(p) has k^2*p free parameters; once that exceeds
        # ~40% of the available observations, OLS is at serious risk of
        # rank-deficiency / near-singular residual covariance.
        method = "lasso" if (k * k * cfg.lags) > 0.4 * n_obs else "ols"

    if method == "lasso":
        res = fit_lasso_var(clean, cfg.lags, cfg.lasso_cv_splits)
        return generalized_fevd(res, horizon=cfg.horizon)

    model = VAR(clean)
    res = None
    for lag in range(cfg.lags, 0, -1):
        trial = model.fit(lag)
        if trial.is_stable(verbose=False):
            res = trial
            break
    if res is None:
        res = model.fit(1)
    return generalized_fevd(res, horizon=cfg.horizon)


def transmission_scores(spill: pd.DataFrame) -> pd.DataFrame:
    arr = spill.to_numpy(dtype=float)
    diag = np.diag(arr)
    to_others = arr.sum(axis=0) - diag
    from_others = arr.sum(axis=1) - diag
    net = to_others - from_others
    tci = float((arr.sum() - diag.sum()) / arr.sum() * 100.0)

    out = pd.DataFrame(
        {
            "to_others": to_others,
            "from_others": from_others,
            "net": net,
        },
        index=spill.index,
    )
    out.attrs["total_connectedness_index"] = tci
    return out.sort_values("net", ascending=False)
