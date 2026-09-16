"""Separating circadian and wake-dependent effects.

Within one wake episode

    DLMO_linear(t) = hours_since_wake(t) + psi_d,   psi_d = wake_d - DLMO (most recent instance)

Both coordinates advance 1 h per hour, so f(phase) and g(time awake) are only
separable if psi varies across days or participants. A molecular anchor removes
the circularity of a sleep-derived anchor but not this collinearity.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ._time import HOUR_NS, circ_mean, circ_sd, utc_ns
from .profile import CircadianProfile


def psi_table(profile: CircadianProfile) -> pd.DataFrame:
    """psi per wake episode: wake time minus the DLMO of the preceding evening (hours)."""
    if not (profile.has_sleep and profile.has_phase):
        raise ValueError(profile._missing("phase angle psi"))
    prep = profile.sleep_table
    inst = profile.dlmo_instants(prep["night_date"].min(), prep["night_date"].max())
    by_date = dict(zip(inst["date"], inst["instant"]))
    rows = []
    for _, r in prep.iterrows():
        i = by_date.get(r["night_date"])
        if i is None:
            continue
        rows.append(dict(night_date=r["night_date"], work_day=r.get("work_day"),
                         psi_wake_h=(r["end"] - i).total_seconds() / 3600,
                         psi_onset_h=(r["onset"] - i).total_seconds() / 3600))
    return pd.DataFrame(rows)


def psi_diagnostics(aligned: pd.DataFrame, phase_col: Optional[str] = None,
                    group: Optional[str] = None, min_sd_h: float = 0.5) -> Dict[str, object]:
    """SD of psi = linear phase - hours_since_wake across rows (optionally by participant).

    min_sd_h is a heuristic threshold, not a validated cut-off.
    """
    if phase_col is None:
        phase_col = "dlmo_phase_h" if "dlmo_phase_h" in aligned.columns else "pdlmo_phase_h"
    df = aligned.dropna(subset=[phase_col, "hours_since_wake"])
    psi = (df[phase_col] - df["hours_since_wake"]) % 24.0
    res = dict(n=len(df), psi_sd_h=circ_sd(psi), psi_mean_h=circ_mean(psi))
    if group is not None:
        g = psi.groupby(df[group])
        res["psi_sd_within_h"] = float(np.nanmean([circ_sd(v) for _, v in g]))
        res["psi_sd_between_h"] = circ_sd([circ_mean(v) for _, v in g])
    res["separation_supported"] = bool(res["psi_sd_h"] >= min_sd_h)
    res["message"] = ("psi varies enough to attempt separating f(phase) and g(time awake)"
                      if res["separation_supported"] else
                      "psi hardly varies: separation of circadian and wake-dependent effects is not supported")
    return res


def concurvity_proxy(aligned: pd.DataFrame, phase_col: str, wake_col: str = "hours_since_wake",
                     harmonics: int = 3) -> float:
    """R^2 of predicting time awake from a Fourier basis of wrapped phase (0 = none, 1 = total).

    A quick proxy only; use mgcv::concurvity on the fitted model for inference.
    """
    df = aligned.dropna(subset=[phase_col, wake_col])
    if len(df) < 2 * harmonics + 2:
        return float("nan")
    ang = 2 * np.pi * (df[phase_col].to_numpy() % 24.0) / 24.0
    X = [np.ones(len(df))]
    for k in range(1, harmonics + 1):
        X += [np.sin(k * ang), np.cos(k * ang)]
    X = np.column_stack(X)
    y = df[wake_col].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(1 - resid.var() / y.var())


def sensitivity_analysis(
    data: pd.DataFrame,
    profile: CircadianProfile,
    analysis: Callable[[pd.DataFrame], Dict[str, float]],
    n: int = 50,
    seed: int = 0,
    **transform_kwargs,
) -> pd.DataFrame:
    """Refit `analysis` with the phase anchor perturbed within its uncertainty.

    The perturbation is a single offset per draw (shared across days), drawn from
    N(0, phase_uncertainty_h^2).
    """
    from copy import deepcopy
    from .transform import transform

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        p = deepcopy(profile)
        delta = rng.normal(0, profile.phase_uncertainty_h)
        p.phase_anchor_h = (p.phase_anchor_h + delta) % 24.0
        if p.trajectory is not None:
            p.trajectory = p.trajectory.copy()
            for c in ("dlmo_h", "dlmo_filter_h"):
                p.trajectory[c] = p.trajectory[c] + delta
        res = analysis(transform(data, p, **transform_kwargs))
        res["draw"], res["delta_h"] = i, delta
        rows.append(res)
    return pd.DataFrame(rows)
