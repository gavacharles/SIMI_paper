from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
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
    px = df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    px = px.where(px > 0)
    out = np.log(px).diff()
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


def fit_spillover_table(returns: pd.DataFrame, cfg: SpilloverConfig) -> pd.DataFrame:
    clean = returns.copy()
    keep = [c for c in clean.columns if clean[c].notna().sum() >= cfg.min_obs]
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

    if clean.shape[1] < 2:
        raise ValueError("Need at least two valid series for spillover estimation.")

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
