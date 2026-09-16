"""Dynamic phase: state-space fusion of phase observations, light model and sleep timing.

Latent state for evening date d (all in hours):

    x_d = [ phi_d,        DLMO of evening d, hours after local midnight of d (unwrapped)
            psi_free,     midsleep - DLMO on nights before a free day
            psi_work ]    midsleep - DLMO on nights before a workday

Process model:     phi_d = phi_(d-1) + drift_d + w_d,   w_d ~ N(0, q_phase^2)
                   drift_d = day-to-day change of light-model DLMO (if available) else 0
                   psi_*   = psi_*(d-1) + N(0, q_psi^2)
Measurement models:
    measured DLMO / pDLMO[assay]   z = phi_d + v,          v ~ N(0, sd_obs^2)
    pDLMO[light model]             z = phi_d + v,          sd inflated by sqrt(n_days) (errors
                                                              are strongly autocorrelated across days)
    sleep midpoint (night d)       z = phi_d + psi_type + v, sd larger on alarm-constrained nights

Sleep enters only through the phase-angle relation with large noise, so a jump in
sleep timing is not read as an equal jump in phase. Innovations are wrapped to
[-12, 12) h (circular measurements). A causal Kalman filter is used for real-time
estimates; a Rauch-Tung-Striebel smoother for retrospective analysis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ._time import fmt_clock, hours_after_local_midnight, wrap12, wrap24
from .phase import PhaseObservation


@dataclass
class StateSpaceConfig:
    q_phase_h: float = 0.25        # SD of unexplained daily phase change (h/day), no light data
    q_phase_light_h: float = 0.15  # SD when light-model drift is available
    max_daily_drift_h: float = 1.5 # cap on light-model drift per day
    q_psi_h: float = 0.05
    # Phase-angle prior. Diffuse by default, so that the *level* of phase comes only from
    # phase observations and sleep informs day-to-day *changes*. A tight prior would
    # silently convert sleep timing into a DLMO estimate, which the spec rules out.
    psi_prior_mean_h: float = 6.0
    psi_prior_sd_h: float = 12.0
    phi_prior_mean_h: float = 21.0   # population prior for the causal filter only
    phi_prior_sd_h: float = 6.0
    r_sleep_free_h: float = 1.0
    r_sleep_work_h: float = 1.5
    light_sd_h: float = 1.1


def _kf_update(x, P, z, H, r):
    y = float(wrap12(z - H @ x))
    S = float(H @ P @ H.T + r ** 2)
    K = (P @ H.T) / S
    x = x + K * y
    P = (np.eye(len(x)) - np.outer(K, H)) @ P
    return x, P


def fuse(
    tz: str,
    observations: Sequence[PhaseObservation],
    sleep_prep: Optional[pd.DataFrame] = None,
    light_dlmo: Optional[pd.DataFrame] = None,
    dates: Optional[Sequence] = None,
    config: Optional[StateSpaceConfig] = None,
    smoother: bool = True,
) -> pd.DataFrame:
    """Run the filter (and optionally smoother). Returns one row per evening date."""
    cfg = config or StateSpaceConfig()
    obs = [o for o in observations if o.dlmo_date is not None]
    all_dates = set(o.dlmo_date for o in obs)
    sleep_by_date: Dict[object, pd.Series] = {}
    if sleep_prep is not None and len(sleep_prep):
        for _, row in sleep_prep.iterrows():
            sleep_by_date[row["night_date"]] = row
        all_dates |= set(sleep_by_date)
    light_by_date: Dict[object, float] = {}
    if light_dlmo is not None and len(light_dlmo):
        light_by_date = dict(zip(light_dlmo["date"], light_dlmo["dlmo_h"]))
        all_dates |= set(light_by_date)
    if dates is not None:
        all_dates |= set(pd.Timestamp(d).date() for d in dates)
    if not all_dates:
        raise ValueError("nothing to fuse")
    d0, d1 = min(all_dates), max(all_dates)
    grid = [d.date() for d in pd.date_range(d0, d1, freq="D")]
    n = len(grid)
    has_work = sleep_prep is not None and len(sleep_prep) and sleep_prep.attrs.get("has_work_info", False)
    n_light = max(len(light_by_date), 1)

    obs_by_date: Dict[object, List[PhaseObservation]] = {}
    for o in obs:
        obs_by_date.setdefault(o.dlmo_date, []).append(o)

    # initial phase prior. The causal filter must not peek at later observations, so it
    # starts from an observation only if one exists on the first date, otherwise from a
    # diffuse population prior (phi_prior_mean_h, SD phi_prior_sd_h).
    first_obs = obs_by_date.get(d0, [])
    if not smoother:
        if first_obs:
            phi0 = float(first_obs[0].dlmo_hours)
        elif light_by_date and d0 in light_by_date:
            phi0 = float(light_by_date[d0])
        else:
            phi0 = cfg.phi_prior_mean_h
    elif obs:
        best = min(obs, key=lambda o: o.effective_sd_h)
        phi0 = float(wrap24(best.dlmo_hours))
        phi0 = phi0 if phi0 > 12 else phi0 + 24.0  # prefer evening representation
    elif light_by_date:
        phi0 = float(np.nanmedian(list(light_by_date.values())))
    else:
        raise ValueError("dynamic phase needs at least one phase observation or light data (Route A has no physiological phase)")

    x = np.array([phi0, cfg.psi_prior_mean_h, cfg.psi_prior_mean_h])
    P = np.diag([cfg.phi_prior_sd_h ** 2, cfg.psi_prior_sd_h ** 2, cfg.psi_prior_sd_h ** 2])
    xs_pred, Ps_pred, xs_filt, Ps_filt, drifts, nobs = [], [], [], [], [], []
    prev_light = None
    for i, d in enumerate(grid):
        # predict
        drift = 0.0
        q = cfg.q_phase_h
        if light_by_date:
            q = cfg.q_phase_light_h
            lv = light_by_date.get(d)
            if lv is not None and prev_light is not None and i > 0:
                drift = float(np.clip(wrap12(lv - prev_light), -cfg.max_daily_drift_h, cfg.max_daily_drift_h))
            if lv is not None:
                prev_light = lv
        if i > 0:
            x = x + np.array([drift, 0.0, 0.0])
            P = P + np.diag([q ** 2, cfg.q_psi_h ** 2, cfg.q_psi_h ** 2])
        xs_pred.append(x.copy()); Ps_pred.append(P.copy()); drifts.append(drift)
        k = 0
        # updates: phase observations
        for o in obs_by_date.get(d, []):
            x, P = _kf_update(x, P, o.dlmo_hours, np.array([1.0, 0.0, 0.0]), o.effective_sd_h); k += 1
        if d in light_by_date:
            x, P = _kf_update(x, P, light_by_date[d], np.array([1.0, 0.0, 0.0]),
                              cfg.light_sd_h * math.sqrt(n_light)); k += 1
        row = sleep_by_date.get(d)
        if row is not None:
            z = hours_after_local_midnight(row["midsleep"], d, tz)
            if has_work and bool(row["work_day"]):
                H, r = np.array([1.0, 0.0, 1.0]), cfg.r_sleep_work_h
            else:
                H = np.array([1.0, 1.0, 0.0])
                r = cfg.r_sleep_free_h if has_work else 0.5 * (cfg.r_sleep_free_h + cfg.r_sleep_work_h)
            if has_work and bool(row.get("alarm", False)) and not bool(row["work_day"]):
                r = cfg.r_sleep_work_h
            x, P = _kf_update(x, P, z, H, r); k += 1
        xs_filt.append(x.copy()); Ps_filt.append(P.copy()); nobs.append(k)

    xs = [v.copy() for v in xs_filt]
    Ps = [v.copy() for v in Ps_filt]
    method = "filter"
    if smoother and n > 1:
        method = "smoother"
        for i in range(n - 2, -1, -1):
            C = Ps_filt[i] @ np.linalg.inv(Ps_pred[i + 1])
            innov = xs[i + 1] - xs_pred[i + 1]
            xs[i] = xs_filt[i] + C @ innov
            Ps[i] = Ps_filt[i] + C @ (Ps[i + 1] - Ps_pred[i + 1]) @ C.T

    out = pd.DataFrame({
        "date": grid,
        "dlmo_h": [v[0] for v in xs],
        "sd_h": [math.sqrt(p[0, 0]) for p in Ps],
        "dlmo_filter_h": [v[0] for v in xs_filt],
        "sd_filter_h": [math.sqrt(p[0, 0]) for p in Ps_filt],
        "psi_free_h": [v[1] for v in xs],
        "psi_work_h": [v[2] for v in xs],
        "light_drift_h": drifts,
        "n_updates": nobs,
    })
    out["dlmo_clock"] = [fmt_clock(h) for h in out["dlmo_h"]]
    out.attrs["method"] = method
    return out
