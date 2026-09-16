"""Scheduling: wake-anchored, phase-anchored, and assay-sampling modes.

For medications the engine only computes requested phase-relative clock times; it
does not determine clinically appropriate dosing schedules.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from ._time import fmt_clock, instant_from_date_hours, parse_clock, to_local_scalar, wrap24
from .profile import CircadianProfile

RULES = ("flag", "drop", "compress")


def _apply_sleep_rule(times_h, wake_h, onset_h, last_before_sleep, rule):
    """times_h: hours after local midnight (can exceed 24). onset_h likewise, > wake_h."""
    limit = onset_h - last_before_sleep
    status = ["ok" if t <= limit else "after_sleep_limit" for t in times_h]
    if rule == "flag" or all(s == "ok" for s in status):
        return list(times_h), status
    if rule == "drop":
        kept = [(t, "ok") for t, s in zip(times_h, status) if s == "ok"]
        return [k[0] for k in kept], [k[1] for k in kept]
    if rule == "compress":
        first, last = times_h[0], times_h[-1]
        if last == wake_h:
            return list(times_h), status
        scale = (limit - wake_h) / (last - wake_h)
        return [wake_h + (t - wake_h) * scale for t in times_h], ["compressed"] * len(times_h)
    raise ValueError(f"rule must be one of {RULES}")


def schedule_wake(
    profile: CircadianProfile,
    offsets: Sequence[float],
    dates: Optional[Sequence] = None,
    mode: str = "predicted",
    wake_time=None,
    predicted_onset=None,
    last_before_sleep: float = 0.0,
    rule: str = "flag",
    next_day_is_workday=None,
    day_types: Optional[dict] = None,
) -> pd.DataFrame:
    """Wake-anchored schedule.

    predicted:      times computed in advance from the profile's typical wake time for the
                    day type; deviations from actual wake must be logged.
    wake_triggered: times computed when wake is logged (pass `wake_time`).
    """
    tz = profile.timezone
    rows = []
    if mode == "wake_triggered":
        if wake_time is None:
            raise ValueError("wake_triggered mode needs the logged wake_time")
        w = to_local_scalar(wake_time, tz)
        d = w.date()
        wake_h = (w - instant_from_date_hours(d, 0, tz)).total_seconds() / 3600
        if predicted_onset is not None:
            on = parse_clock(predicted_onset)
        elif profile.has_sleep and profile.sleep.typical:
            key = "workday" if (next_day_is_workday if next_day_is_workday is not None else pd.Timestamp(d + pd.Timedelta(days=1)).dayofweek < 5) else "free"
            on = profile.sleep.typical.get(key, {}).get("onset_before", float("nan"))
        else:
            on = float("nan")
        on_h = on + 24.0 if np.isfinite(on) and on < 12 else on
        times = [wake_h + o for o in offsets]
        if np.isfinite(on_h):
            times, status = _apply_sleep_rule(times, wake_h, on_h, last_before_sleep, rule)
        else:
            status = ["no_sleep_prediction"] * len(times)
        for i, (t, s) in enumerate(zip(times, status)):
            inst = instant_from_date_hours(d, t, tz)
            rows.append(dict(date=d, event=i + 1, anchor="wake", anchor_time=fmt_clock(wake_h), time=inst,
                             clock=fmt_clock(t), status=s, mode=mode))
        return pd.DataFrame(rows)

    if mode != "predicted":
        raise ValueError("mode must be 'predicted' or 'wake_triggered'")
    if not profile.has_sleep or not profile.sleep.typical:
        raise ValueError(profile._missing("wake-anchored scheduling (needs sleep records with work/free-day information)"))
    if dates is None:
        raise ValueError("predicted mode needs dates")
    def _type(day):
        if day_types is not None and day in day_types:
            return day_types[day]
        return "workday" if pd.Timestamp(day).dayofweek < 5 else "free"

    for d in [pd.Timestamp(x).date() for x in dates]:
        dtype = _type(d)
        ntype = _type(d + pd.Timedelta(days=1))
        wake_h = profile.sleep.typical[dtype]["wake"]
        on = profile.sleep.typical[ntype]["onset_before"]
        on_h = on + 24.0 if on < 12 else on
        times, status = _apply_sleep_rule([wake_h + o for o in offsets], wake_h, on_h, last_before_sleep, rule)
        for i, (t, s) in enumerate(zip(times, status)):
            rows.append(dict(date=d, day_type=dtype, event=i + 1, anchor="wake", anchor_time=fmt_clock(wake_h),
                             time=instant_from_date_hours(d, t, tz), clock=fmt_clock(t),
                             predicted_onset=fmt_clock(on), status=s, mode=mode))
    out = pd.DataFrame(rows)
    out.attrs["note"] = "predicted wake times; log deviations between predicted and actual wake"
    return out


def schedule_constrained(wake, predicted_onset, events: int, first_after: float,
                         last_before_sleep: float, min_spacing: float) -> dict:
    """Evenly spaced events between wake+first_after and onset-last_before_sleep."""
    w, on = parse_clock(wake), parse_clock(predicted_onset)
    if on < w:
        on += 24.0
    a, b = w + first_after, on - last_before_sleep
    if events == 1:
        times = [a]
    else:
        times = list(np.linspace(a, b, events))
    spacing = (b - a) / (events - 1) if events > 1 else float("inf")
    ok = spacing >= min_spacing and b >= a
    min_onset = w + first_after + min_spacing * (events - 1) + last_before_sleep
    return dict(times=[fmt_clock(t) for t in times], spacing_h=spacing, valid=bool(ok),
                valid_if_onset_at_or_after=fmt_clock(min_onset))


def schedule_phase(
    profile: CircadianProfile,
    offsets: Sequence[float],
    dates: Sequence,
    z: float = 1.96,
) -> pd.DataFrame:
    """Phase-anchored schedule: DLMO_d + offset. Anchor uncertainty propagates 1:1."""
    if not profile.has_phase:
        raise ValueError(profile._missing("phase-anchored scheduling"))
    tz = profile.timezone
    rows = []
    for d in [pd.Timestamp(x).date() for x in dates]:
        h, sd = profile.dlmo_for_date(d)
        for i, o in enumerate(offsets):
            t = h + o
            rows.append(dict(anchor_date=d, event=i + 1, offset_h=o, anchor_source=profile.phase_anchor_type,
                             anchor_clock=fmt_clock(h), time=instant_from_date_hours(d, t, tz), clock=fmt_clock(t),
                             lower=fmt_clock(t - z * sd), upper=fmt_clock(t + z * sd), sd_h=round(sd, 2),
                             phase_mode=profile.phase_mode))
    out = pd.DataFrame(rows)
    out.attrs["note"] = ("scheduled clock times inherit the anchor's individual uncertainty; "
                         "the engine does not determine clinically appropriate dosing schedules")
    return out


def schedule_phase_sample(
    assay: str = "HairTime",
    prior: Optional[CircadianProfile] = None,
    window_after_dlmo: tuple = (9.0, 17.0),
    prior_range=("19:00", "23:00"),
    phase_angle_midsleep_h: float = 6.0,
    phase_angle_sd_h: float = 1.0,
    coverage_sd: float = 2.0,
) -> dict:
    """Choose a sampling time inside the assay's validated window, given a DLMO prior.

    Prior, in order of preference: physiological phase in `prior`; sleep-based prior
    MSFsc - phase_angle_midsleep_h (SD phase_angle_sd_h, a population assumption);
    otherwise the population range `prior_range`.
    """
    lo_w, hi_w = window_after_dlmo
    if prior is not None and prior.has_phase:
        mean, sd = prior.phase_anchor_h, prior.phase_uncertainty_h
        kind = f"{prior.phase_anchor_type} {fmt_clock(mean)} (SD {sd:.2f} h)"
    elif prior is not None and np.isfinite(prior.behavioural_anchor_h):
        mean, sd = wrap24(prior.behavioural_anchor_h - phase_angle_midsleep_h), phase_angle_sd_h
        kind = (f"sleep-based: MSFsc {prior.behavioural_anchor} - {phase_angle_midsleep_h:g} h "
                f"(SD {sd:g} h, population phase-angle assumption)")
    else:
        mean = sd = None
        kind = f"population range {prior_range[0]}-{prior_range[1]}"

    if mean is None:
        a, b = parse_clock(prior_range[0]), parse_clock(prior_range[1])
        if b < a:
            b += 24
        start, end = b + lo_w, a + hi_w
        rec = (start + end) / 2 if end >= start else None
        prob = 1.0 if end >= start else 0.0
    else:
        m = mean if mean >= 12 else mean + 24
        start, end = m + coverage_sd * sd + lo_w, m - coverage_sd * sd + hi_w
        # maximise P(lo <= t - DLMO <= hi) under N(m, sd^2): optimum at the window centre
        rec = m + (lo_w + hi_w) / 2
        prob = float(norm.cdf((rec - lo_w - m) / sd) - norm.cdf((rec - hi_w - m) / sd))
    robust = (fmt_clock(start), fmt_clock(end)) if end >= start else None
    res = dict(
        assay=assay, prior=kind, validated_window_after_dlmo=window_after_dlmo,
        robust_window=robust, recommended_time=fmt_clock(rec) if rec is not None else None,
        prob_in_window=round(prob, 3),
        note="record exact sampling time, timezone and day type; avoid evening and night samples",
    )
    if robust is None:
        res["warning"] = "no clock window lies inside the validated window for the whole prior range; narrow the prior (e.g., sleep records)"
    return res
