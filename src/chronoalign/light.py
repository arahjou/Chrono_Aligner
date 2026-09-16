"""Light-driven circadian model: model-predicted DLMO (pDLMO[light_model]).

Implements the simplified Kronauer model of Forger, Jewett & Kronauer (1999),
with the parameterisation used in the open-source `circadian` package
(Arcascope). DLMO is taken as CBTmin - 7 h, where CBTmin is the minimum of x.

This is a research-grade implementation. Its individual accuracy is not
established by this package; Woelders et al. (2017) reported SD 1.1 h for a
Kronauer-type model with 9 days of light and activity in a small sample.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ._time import evening_date, hours_after_local_midnight, to_local, utc_ns, HOUR_NS


@dataclass
class Forger99Params:
    taux: float = 24.2
    mu: float = 0.23
    G: float = 33.75
    alpha_0: float = 0.05
    beta: float = 0.0075
    p: float = 0.5
    I0: float = 9500.0
    k: float = 0.55
    cbt_to_dlmo: float = 7.0


def _deriv(state, light, prm: Forger99Params):
    x, xc, n = state
    alpha = prm.alpha_0 * (max(light, 0.0) / prm.I0) ** prm.p
    bhat = prm.G * (1.0 - n) * alpha * (1 - 0.4 * x) * (1 - 0.4 * xc)
    mu_term = prm.mu * (xc - 4.0 / 3.0 * xc ** 3)
    taux_term = (24.0 / (0.99669 * prm.taux)) ** 2 + prm.k * bhat
    return np.array([
        math.pi / 12.0 * (xc + bhat),
        math.pi / 12.0 * (mu_term - x * taux_term),
        60.0 * (alpha * (1.0 - n) - prm.beta * n),
    ])


def integrate_forger99(lux: np.ndarray, dt_h: float, state0=None, prm: Optional[Forger99Params] = None) -> np.ndarray:
    """RK4 integration with piecewise-constant light. Returns states at each grid point."""
    prm = prm or Forger99Params()
    s = np.array([-0.0843259, -1.09607546, 0.45584306]) if state0 is None else np.asarray(state0, float)
    out = np.empty((len(lux), 3))
    for i, L in enumerate(lux):
        out[i] = s
        k1 = _deriv(s, L, prm)
        k2 = _deriv(s + 0.5 * dt_h * k1, L, prm)
        k3 = _deriv(s + 0.5 * dt_h * k2, L, prm)
        k4 = _deriv(s + dt_h * k3, L, prm)
        s = s + dt_h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def _minima(t: np.ndarray, x: np.ndarray, min_sep_h: float = 18.0):
    idx = [i for i in range(1, len(x) - 1) if x[i] <= x[i - 1] and x[i] < x[i + 1]]
    keep = []
    for i in idx:
        if keep and t[i] - t[keep[-1]] < min_sep_h:
            if x[i] < x[keep[-1]]:
                keep[-1] = i
        else:
            keep.append(i)
    out = []
    for i in keep:  # parabolic refinement
        y0, y1, y2 = x[i - 1], x[i], x[i + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        out.append(t[i] + off * (t[1] - t[0]))
    return np.array(out)


def light_model_dlmo(
    light: pd.DataFrame,
    tz: str,
    timestamp: str = "datetime",
    lux: str = "lux",
    dt_min: float = 6.0,
    burn_in_loops: int = 10,
    params: Optional[Forger99Params] = None,
) -> pd.DataFrame:
    """Predict daily DLMO from a light time series.

    Returns a DataFrame with columns date (evening date), dlmo_h (hours after local
    midnight of that date), dlmo_instant.
    """
    ts = to_local(light[timestamp], tz)
    ns = utc_ns(ts)
    t0 = ns.min()
    th = (ns - t0) / HOUR_NS
    dt_h = dt_min / 60.0
    grid = np.arange(0.0, th.max() + 1e-9, dt_h)
    bins = np.floor(th / dt_h + 1e-6).astype(int)
    sums = np.bincount(bins, weights=light[lux].to_numpy(float), minlength=len(grid))[: len(grid)]
    cnts = np.bincount(bins, minlength=len(grid))[: len(grid)]
    lux_grid = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    miss = np.isnan(lux_grid).mean()
    if miss > 0.1:
        warnings.warn(f"{miss:.0%} of light bins missing; treated as darkness")
    lux_grid = np.nan_to_num(lux_grid, nan=0.0)
    days = th.max() / 24.0
    if days < 9:
        warnings.warn(f"light model with {days:.1f} days of data (validated with >= ~9 days)")

    # burn-in: loop the first 7 days (or all data if shorter) to reach entrainment
    n_week = int(min(len(grid), round(7 * 24 / dt_h)))
    n_week -= n_week % int(round(24 / dt_h)) or 0
    n_week = max(n_week, int(round(24 / dt_h)))
    state = None
    for _ in range(burn_in_loops):
        st = integrate_forger99(lux_grid[:n_week], dt_h, state, params)
        state = st[-1]
    states = integrate_forger99(lux_grid, dt_h, state, params)
    prm = params or Forger99Params()
    cbt = _minima(grid, states[:, 0])
    rows = []
    for c in cbt:
        inst = (pd.Timestamp(t0, tz="UTC") + pd.Timedelta(hours=float(c - prm.cbt_to_dlmo))).tz_convert(tz)
        d = evening_date(inst)
        rows.append(dict(date=d, dlmo_h=hours_after_local_midnight(inst, d, tz), dlmo_instant=inst))
    out = pd.DataFrame(rows)
    # first/last markers can be edge artefacts of the grid
    return out.iloc[1:-1].reset_index(drop=True) if len(out) > 3 else out
