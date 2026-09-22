from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from statsmodels.tsa.stattools import grangercausalitytests


CorrelationType = Literal["pearson", "spearman", "partial"]


@dataclass
class NetworkConfig:
    correlation_type: CorrelationType = "partial"
    threshold: float = 0.2


def correlation_matrix(data: pd.DataFrame, correlation_type: CorrelationType = "partial") -> pd.DataFrame:
    x = data.dropna().copy()
    if x.empty:
        raise ValueError("No valid observations available for correlation matrix.")
    if x.shape[0] < x.shape[1]:
        raise ValueError(
            f"Correlation matrix needs at least as many rows as columns "
            f"(got {x.shape[0]} rows, {x.shape[1]} columns); reduce the node set or widen the window."
        )

    if correlation_type == "pearson":
        return x.corr(method="pearson")
    if correlation_type == "spearman":
        return x.corr(method="spearman")
    if correlation_type == "partial":
        # LedoitWolf shrinkage is applied precisely because the raw sample
        # covariance is ill-conditioned in this regime (k close to n). scipy's
        # pinvh internally computes 1/eigenvalue for *all* eigenvalues before
        # masking out the ones below its cutoff, which raises a spurious
        # divide-by-zero/invalid RuntimeWarning even though the masked final
        # result is valid -- suppress that known false alarm here, but still
        # assert finiteness of the actual output rather than trust it blindly.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            lw = LedoitWolf().fit(x.values)
            precision = lw.precision_
            d = np.sqrt(np.diag(precision))
            partial = -precision / np.outer(d, d)
        np.fill_diagonal(partial, 1.0)
        if not np.all(np.isfinite(partial)):
            raise FloatingPointError("Partial correlation matrix contains non-finite values.")
        return pd.DataFrame(partial, index=x.columns, columns=x.columns)

    raise ValueError(f"Unsupported correlation_type={correlation_type}")


def build_correlation_network(corr: pd.DataFrame, threshold: float = 0.2) -> nx.Graph:
    g = nx.Graph()
    cols = list(corr.columns)
    g.add_nodes_from(cols)

    for i, src in enumerate(cols):
        for j in range(i + 1, len(cols)):
            dst = cols[j]
            w = float(corr.loc[src, dst])
            if np.abs(w) >= threshold:
                g.add_edge(src, dst, weight=w, abs_weight=abs(w), distance=1.0 - abs(w))
    return g


def build_spillover_network(spill: pd.DataFrame, threshold: float = 2.0) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(spill.columns)
    for recv in spill.index:
        for src in spill.columns:
            if recv == src:
                continue
            w = float(spill.loc[recv, src])
            if w >= threshold:
                g.add_edge(src, recv, weight=w)
    return g


def build_granger_network(
    series: pd.DataFrame,
    maxlag: int = 6,
    alpha: float = 0.05,
    test: str = "ssr_chi2test",
) -> nx.DiGraph:
    data = series.dropna().copy()
    if data.shape[1] < 2:
        raise ValueError("Need at least two series for Granger network.")

    g = nx.DiGraph()
    g.add_nodes_from(data.columns)

    for cause in data.columns:
        for effect in data.columns:
            if cause == effect:
                continue
            pair = data[[effect, cause]].dropna()
            if len(pair) < (maxlag + 10):
                continue
            try:
                res = grangercausalitytests(pair, maxlag=maxlag, verbose=False)
                pvals = [(lag, out[0][test][1]) for lag, out in res.items()]
                lag_star, p_star = min(pvals, key=lambda x: x[1])
                if p_star <= alpha:
                    g.add_edge(cause, effect, weight=1.0 - float(p_star), lag=int(lag_star), pvalue=float(p_star))
            except Exception:
                continue
    return g


def minimum_spanning_tree(corr: pd.DataFrame) -> nx.Graph:
    base = nx.Graph()
    cols = list(corr.columns)
    base.add_nodes_from(cols)

    for i, src in enumerate(cols):
        for j in range(i + 1, len(cols)):
            dst = cols[j]
            w = float(corr.loc[src, dst])
            dist = 1.0 - abs(w)
            base.add_edge(src, dst, weight=w, distance=dist)

    return nx.minimum_spanning_tree(base, weight="distance")
