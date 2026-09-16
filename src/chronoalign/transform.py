"""Add personalised temporal coordinates to arbitrary time series.

The original timestamps are never changed; columns are only added.

    crt_phase_h     = (t - MSFsc) mod 24                 behavioural, [0, 24)
    crt_linear_h    = t - MSFsc_k                        unbounded, UTC, reference instance k
    (p)dlmo_phase_h = (t - DLMO_d) mod 24                physiological, most recent DLMO instance
    (p)dlmo_linear_h= t - DLMO_k                         unbounded, UTC
    hours_since_wake, prior_sleep_duration_h             from the sleep table
    daily_sleep_phase_h                                  time since smoothed day-specific midsleep
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._time import HOUR_NS, clock_hours, instant_from_date_hours, to_local, unwrap_sequence, utc_ns, wrap24
from .profile import CircadianProfile

VALID_REFS = {"chronotype", "phase", "wake", "daily_sleep"}


def _daily_sleep_reference(prep: pd.DataFrame, tz: str, alpha: float = 0.3, alarm_weight: float = 0.5) -> pd.DataFrame:
    """Causal exponentially weighted day-specific midsleep (behavioural, NOT circadian phase).

    Alarm-constrained nights get a smaller update weight.
    """
    ms_h = np.array([
        (row["midsleep"] - instant_from_date_hours(row["night_date"], 0.0, tz)).total_seconds() / 3600
        for _, row in prep.iterrows()
    ])
    ms_h = unwrap_sequence(ms_h)
    ref = np.empty_like(ms_h)
    alarms = prep["alarm"].to_numpy() if "alarm" in prep.columns else np.zeros(len(prep), bool)
    for i, v in enumerate(ms_h):
        if i == 0:
            ref[i] = v
        else:
            a = alpha * (alarm_weight if alarms[i] else 1.0)
            ref[i] = a * v + (1 - a) * ref[i - 1]
    inst = [instant_from_date_hours(d, h, tz) for d, h in zip(prep["night_date"], ref)]
    return pd.DataFrame({"night_date": prep["night_date"], "ref_h": ref, "instant": inst})


def transform(
    data: pd.DataFrame,
    profile: CircadianProfile,
    timestamp: str = "datetime",
    reference: Union[str, Sequence[str]] = ("phase", "chronotype", "wake"),
    reference_date=None,
    circular_encoding: bool = False,
    max_hours_awake: float = 30.0,
    event_times: Optional[Iterable] = None,
    event_name: str = "event",
) -> pd.DataFrame:
    """Return a copy of `data` with coordinate columns added.

    reference_date: evening date whose anchor instance is the reference k for the linear
    coordinates (default: the latest anchor instance at or before the first timestamp).
    event_times: optional timestamps (e.g. meals) -> hours_since_<event_name>.
    """
    refs = [reference] if isinstance(reference, str) else list(reference)
    bad = set(refs) - VALID_REFS
    if bad:
        raise ValueError(f"unknown reference(s) {bad}; choose from {sorted(VALID_REFS)}")
    tz = profile.timezone
    out = data.copy()
    ts = to_local(out[timestamp], tz)
    t_ns = utc_ns(ts)
    out["clock_time_h"] = clock_hours(ts)
    explicit_ref = reference_date is not None
    ref_day = pd.Timestamp(reference_date).date() if explicit_ref else ts.min().date()
    t_min = ts.min()

    def _reference_instant(hours_for_date):
        """Explicit: the instance belonging to evening `reference_date`.
        Default: the latest instance at or before the first timestamp."""
        if explicit_ref:
            return instant_from_date_hours(ref_day, hours_for_date(ref_day), tz)
        cands = []
        for k in (-2, -1, 0):
            d = ref_day + pd.Timedelta(days=k)
            cands.append(instant_from_date_hours(d, hours_for_date(d), tz))
        before = [c for c in cands if c <= t_min]
        return max(before) if before else min(cands)

    need_sleep = {"wake", "daily_sleep"} & set(refs)
    if need_sleep and not profile.has_sleep:
        raise ValueError(profile._missing("wake-dependent coordinates"))

    if "wake" in refs:
        prep = profile.sleep_table
        on, end = utc_ns(prep["onset"]), utc_ns(prep["end"])
        dur = prep["duration_h"].to_numpy()
        j = np.searchsorted(on, t_ns, side="right") - 1
        asleep = (j >= 0) & (t_ns < end[np.clip(j, 0, None)])
        k = np.searchsorted(end, t_ns, side="right") - 1
        hsw = np.where(k >= 0, (t_ns - end[np.clip(k, 0, None)]) / HOUR_NS, np.nan)
        psd = np.where(k >= 0, dur[np.clip(k, 0, None)], np.nan)
        stale = hsw > max_hours_awake
        hsw = np.where(asleep | stale, np.nan, hsw)
        psd = np.where(asleep | stale, np.nan, psd)
        out["asleep"] = asleep
        out["hours_since_wake"] = hsw
        out["prior_sleep_duration_h"] = psd
        if stale.any():
            out.attrs.setdefault("warnings", []).append(
                f"{int(stale.sum())} rows > {max_hours_awake} h after the last recorded wake (missing sleep record?) set to NaN")

    if "chronotype" in refs:
        if not np.isfinite(profile.behavioural_anchor_h):
            raise ValueError("chronotype-relative time needs MSFsc, which is not available: "
                             + "; ".join(f for f in profile.flags if f.startswith("MSFsc")) or profile._missing("CRT"))
        phi = profile.behavioural_anchor_h
        out["crt_phase_h"] = wrap24(out["clock_time_h"].to_numpy() - phi)
        # reference instance: the anchor time within the night that starts on ref_day
        k_inst = _reference_instant(lambda d: phi if phi >= 12 else phi + 24.0)
        out["crt_linear_h"] = (t_ns - k_inst.value) / HOUR_NS
        out.attrs["crt_reference_instant"] = k_inst
        if circular_encoding:
            out["crt_sin"] = np.sin(2 * math.pi * out["crt_phase_h"] / 24)
            out["crt_cos"] = np.cos(2 * math.pi * out["crt_phase_h"] / 24)

    if "phase" in refs:
        if not profile.has_phase:
            raise ValueError(profile._missing("physiological phase"))
        prefix = "dlmo" if profile.phase_is_measured else "pdlmo"
        start = (ts.min() - pd.Timedelta(days=2)).date()
        stop = (ts.max() + pd.Timedelta(days=1)).date()
        inst = profile.dlmo_instants(start, stop)
        i_ns = utc_ns(inst["instant"])
        order = np.argsort(i_ns)
        i_ns, sds = i_ns[order], inst["sd_h"].to_numpy()[order]
        k = np.clip(np.searchsorted(i_ns, t_ns, side="right") - 1, 0, None)
        out[f"{prefix}_phase_h"] = wrap24((t_ns - i_ns[k]) / HOUR_NS)
        k_ref = _reference_instant(lambda d: profile.dlmo_for_date(d)[0])
        out[f"{prefix}_linear_h"] = (t_ns - k_ref.value) / HOUR_NS
        out.attrs["phase_reference_instant"] = k_ref
        out["phase_uncertainty_h"] = sds[k]
        out["anchor_source"] = profile.phase_anchor_type
        out["phase_mode"] = profile.phase_mode if profile.phase_mode != "fixed" else "fixed (stability assumption)"
        if circular_encoding:
            out[f"{prefix}_sin"] = np.sin(2 * math.pi * out[f"{prefix}_phase_h"] / 24)
            out[f"{prefix}_cos"] = np.cos(2 * math.pi * out[f"{prefix}_phase_h"] / 24)

    if "daily_sleep" in refs:
        ref = _daily_sleep_reference(profile.sleep_table, tz)
        r_ns = utc_ns(ref["instant"])
        k = np.searchsorted(r_ns, t_ns, side="right") - 1
        out["daily_sleep_phase_h"] = np.where(k >= 0, (t_ns - r_ns[np.clip(k, 0, None)]) / HOUR_NS, np.nan)
        out.attrs.setdefault("warnings", []).append(
            "daily_sleep_phase_h is nearly collinear with hours_since_wake; do not enter both as predictors")

    if event_times is not None:
        ev = np.sort(utc_ns(to_local(list(event_times), tz)))
        k = np.searchsorted(ev, t_ns, side="right") - 1
        out[f"hours_since_{event_name}"] = np.where(k >= 0, (t_ns - ev[np.clip(k, 0, None)]) / HOUR_NS, np.nan)
    return out
