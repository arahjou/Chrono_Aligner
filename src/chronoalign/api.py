"""Top-level API: fit / transform / schedule / schedule_phase_sample."""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ._time import fmt_clock, hours_after_local_midnight, instant_from_date_hours, wrap24
from .grades import assign_grade, route as route_of
from .light import light_model_dlmo
from .phase import PhaseObservation, QCPolicy, combine, disagreements, SOURCES
from .profile import CircadianProfile
from .sleep import sleep_phenotype
from .statespace import StateSpaceConfig, fuse


def fit(
    sleep: Optional[pd.DataFrame] = None,
    light: Optional[pd.DataFrame] = None,
    phase_observations: Optional[Sequence[PhaseObservation]] = None,
    tz: str = "Europe/Berlin",
    dynamic: bool = False,
    smoother: bool = True,
    min_days: int = 7,
    min_free_days: int = 2,
    preferred_days: int = 14,
    schedule_type: Optional[str] = None,
    pool_observations: bool = True,
    qc_policy: Optional[QCPolicy] = None,
    statespace_config: Optional[StateSpaceConfig] = None,
    disagreement_z: float = 1.96,
    light_kwargs: Optional[dict] = None,
) -> CircadianProfile:
    """Build a CircadianProfile from any combination of the three input routes."""
    obs = list(phase_observations or [])
    if sleep is None and light is None and not obs:
        raise ValueError("give at least one of: sleep, light, phase_observations")

    ph, prep = (None, None)
    stype = schedule_type or "regular"
    if sleep is not None:
        ph, prep = sleep_phenotype(sleep, tz, min_days, min_free_days, preferred_days, schedule_type)
        stype = ph.schedule_type

    light_dlmo = None
    if light is not None:
        light_dlmo = light_model_dlmo(light, tz, **(light_kwargs or {}))

    for o in obs:
        o.resolve(tz, qc_policy, stype)

    has_phase = bool(obs) or light_dlmo is not None and len(light_dlmo) > 0
    rte = route_of(prep is not None, has_phase)
    grade = assign_grade(prep is not None, bool(ph and ph.has_work_info), light_dlmo is not None,
                         {o.source for o in obs})
    prof = CircadianProfile(timezone=tz, route=rte, evidence_grade=grade, schedule_type=stype,
                            observations=obs, light_dlmo=light_dlmo)

    if ph is not None:
        prof.sleep, prof.sleep_table = ph, prep
        prof.n_days_sleep, prof.n_free_days = ph.n_days, ph.n_free
        prof.social_jetlag_h, prof.sri = ph.social_jetlag_h, ph.sri
        if np.isfinite(ph.msfsc):
            prof.behavioural_anchor, prof.behavioural_anchor_h = fmt_clock(ph.msfsc), ph.msfsc
            prof.behavioural_se_min = ph.msfsc_se_min
        elif ph.msfsc_note:
            prof.flags.append("MSFsc: " + ph.msfsc_note)
        prof.flags += ph.warnings

    # ---------------- physiological anchor (fixed) ----------------
    if obs:
        best_tier = min(o.tier for o in obs)
        tier_obs = [o for o in obs if o.tier == best_tier]
        comb = combine(tier_obs, independent=pool_observations)
        prof.phase_anchor_h, prof.phase_uncertainty_h = comb.hours, comb.sd_h
        prof.phase_anchor, prof.phase_anchor_type = fmt_clock(comb.hours), comb.label
        ref = min(tier_obs, key=lambda o: o.effective_sd_h)
        anchor_date = ref.dlmo_date
        prof.phase_anchor_date = f"{anchor_date} (evening)" + (f"; sample day type: {ref.day_type}" if ref.day_type else "")
        prof.flags += comb.flags
        lower = [o.label for o in obs if o.tier > best_tier]
        if lower:
            prof.flags.append(f"fixed anchor uses {comb.label}; lower-validity sources kept as observations: {sorted(set(lower))}")
        prof.disagreements = disagreements(obs, disagreement_z)
    elif light_dlmo is not None and len(light_dlmo):
        h = np.asarray(light_dlmo["dlmo_h"], float)
        med = float(np.median(h))
        prof.phase_anchor_h = float(wrap24(med))
        prof.phase_anchor, prof.phase_anchor_type = fmt_clock(med), "pDLMO[light_model]"
        prof.phase_uncertainty_h = SOURCES["light_model"]["base_sd_h"]
        anchor_date = light_dlmo["date"].iloc[len(light_dlmo) // 2]
        prof.phase_anchor_date = f"median over {light_dlmo['date'].iloc[0]} .. {light_dlmo['date'].iloc[-1]}"
    else:
        anchor_date = None

    if prof.has_phase:
        prof.phase_mode = "fixed"
        if prep is not None:
            span = f"{prep['night_date'].min()} .. {prep['night_date'].max()}"
        else:
            span = f"sampling evening {anchor_date} only"
        prof.stability_assumption = f"phase constant over {span}"
        if prep is None:
            prof.flags.append("Route B: applying the anchor to other days is an assumption of stability")

    # ---------------- dynamic phase ----------------
    if dynamic:
        if not prof.has_phase:
            raise ValueError(prof._missing("dynamic phase"))
        tr = fuse(tz, obs, prep, light_dlmo, config=statespace_config, smoother=smoother)
        prof.trajectory = tr
        prof.phase_mode = tr.attrs["method"]
        prof.stability_assumption = None
        if prep is None and light_dlmo is None:
            prof.flags.append("dynamic mode without sleep or light data: phase is held constant, only uncertainty grows")
        if not smoother:
            prof.flags.append("causal filter (real-time); estimates use past data only")

    # ---------------- phase angle on the anchor evening ----------------
    if prof.has_phase and prep is not None and anchor_date is not None:
        night = prep[prep["night_date"] == anchor_date]
        if len(night):
            row = night.iloc[0]
            dlmo_h = next((o.dlmo_hours for o in obs if o.dlmo_date == anchor_date and o.tier == min(x.tier for x in obs)),
                          None) if obs else None
            if dlmo_h is None:
                dlmo_h, _ = prof.dlmo_for_date(anchor_date)
            inst = instant_from_date_hours(anchor_date, dlmo_h, tz)
            prof.phase_angle_onset_h = (row["onset"] - inst).total_seconds() / 3600
            prof.phase_angle_midsleep_h = (row["midsleep"] - inst).total_seconds() / 3600
            prof.phase_angle_wake_h = (row["end"] - inst).total_seconds() / 3600
        else:
            prof.flags.append(f"no sleep episode recorded on the anchor evening {anchor_date}: phase angle not computed")

    if stype == "shift":
        prof.flags.append("shift work: sleep-only phase estimates are least valid here; molecular anchors not yet validated against DLMO in this group")
    return prof
