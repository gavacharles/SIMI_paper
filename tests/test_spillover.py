from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.volatility.spillover import SpilloverConfig, fit_spillover_table, generalized_fevd, transmission_scores


def test_generalized_fevd_benchmark_values() -> None:
    """
    DY-style 2-variable VAR(1) benchmark fixture.
    The expected FEVD values are frozen for regression testing.
    """
    a = np.array([[[0.50, 0.20], [0.10, 0.40]]])
    sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
    res = SimpleNamespace(neqs=2, k_ar=1, coefs=a, sigma_u=sigma, names=["x1", "x2"])

    spill = generalized_fevd(res, horizon=10)

    expected = pd.DataFrame(
        [[81.780093, 18.219907], [12.137214, 87.862786]],
        index=["x1", "x2"],
        columns=["x1", "x2"],
    )

    np.testing.assert_allclose(spill.values, expected.values, atol=1e-6)


def test_spillover_rows_sum_to_100() -> None:
    rng = np.random.default_rng(42)
    n = 160
    e = rng.normal(size=(n, 3))
    x = np.zeros_like(e)
    for t in range(1, n):
        x[t, 0] = 0.55 * x[t - 1, 0] + 0.20 * x[t - 1, 1] + e[t, 0]
        x[t, 1] = 0.10 * x[t - 1, 0] + 0.60 * x[t - 1, 1] + 0.10 * x[t - 1, 2] + e[t, 1]
        x[t, 2] = 0.05 * x[t - 1, 1] + 0.45 * x[t - 1, 2] + e[t, 2]

    ret = pd.DataFrame(x, columns=["a", "b", "c"])
    spill = fit_spillover_table(ret, SpilloverConfig(lags=1, horizon=8, min_obs=80))
    row_sums = spill.sum(axis=1)

    np.testing.assert_allclose(row_sums.values, np.full(len(row_sums), 100.0), atol=1e-6)

    tx = transmission_scores(spill)
    assert {"to_others", "from_others", "net"}.issubset(tx.columns)
