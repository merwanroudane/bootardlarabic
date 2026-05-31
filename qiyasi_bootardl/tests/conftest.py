"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def cointegrated_data() -> pd.DataFrame:
    """Synthetic cointegrated I(1) system: y adjusts toward 1 + 0.6 x1 + 0.3 x2."""
    rng = np.random.default_rng(20240501)
    n = 120
    x1 = np.cumsum(rng.normal(0, 1, n)) + 50
    x2 = np.cumsum(rng.normal(0, 1, n)) + 30
    eq = 1.0 + 0.6 * x1 + 0.3 * x2
    y = np.empty(n)
    y[0] = eq[0] + rng.normal(0, 1)
    for t in range(1, n):
        y[t] = y[t - 1] + 0.5 * (eq[t] - y[t - 1]) + 0.4 * (x1[t] - x1[t - 1]) + rng.normal(0, 0.8)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


@pytest.fixture(scope="session")
def independent_data() -> pd.DataFrame:
    """Three independent random walks (no cointegration)."""
    rng = np.random.default_rng(99)
    n = 120
    return pd.DataFrame(
        {
            "y": np.cumsum(rng.normal(0, 1, n)),
            "x1": np.cumsum(rng.normal(0, 1, n)),
            "x2": np.cumsum(rng.normal(0, 1, n)),
        }
    )
