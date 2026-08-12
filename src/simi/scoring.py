from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


@dataclass
class SIMIWeights:
    spillover: float = 0.5
    centrality: float = 0.5


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def estimate_cipi_basket_shares(
    panel: pd.DataFrame,
    material_columns: list[str],
    target_col: str = "CIPI_MAT",
) -> pd.Series:
    if len(material_columns) == 0:
        return pd.Series(dtype=float)

    if target_col not in panel.columns:
        return pd.Series(1.0 / len(material_columns), index=material_columns)

    x = panel[material_columns].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(panel[target_col], errors="coerce")
    keep = x.notna().all(axis=1) & y.notna()
    x = x.loc[keep]
    y = y.loc[keep]

    if x.empty:
        return pd.Series(1.0 / len(material_columns), index=material_columns)

    x_std = x / x.mean(axis=0).replace(0, np.nan)
    x_std = x_std.replace([np.inf, -np.inf], np.nan).dropna()
    y_std = y.loc[x_std.index]
    if x_std.empty:
        return pd.Series(1.0 / len(material_columns), index=material_columns)

    try:
        res = lsq_linear(
            x_std.values,
            y_std.values,
            bounds=(0.0, 1.0),
            max_iter=5000,
            lsmr_tol="auto",
        )
        w = res.x
    except Exception:
        return pd.Series(1.0 / len(material_columns), index=material_columns)

    if (not np.all(np.isfinite(w))) or w.sum() <= 0:
        return pd.Series(1.0 / len(material_columns), index=material_columns)

    w = w / w.sum()
    return pd.Series(w, index=material_columns)


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
