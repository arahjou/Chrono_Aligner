"""Clock-time and circular helpers.

Conventions used throughout ChronoAlign
---------------------------------------
* Clock times are floats in hours, in [0, 24).
* Signed differences between clock times are wrapped to [-12, 12).
* All durations are computed on timezone-aware datetimes (UTC arithmetic), so
  DST transitions and travel never add spurious 1-h errors.
* Summaries of clock times use circular statistics (the mean of 23:30 and
  00:30 is 00:00, not 12:00).
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
import pandas as pd

HOUR_NS = 3_600_000_000_000
TWO_PI = 2.0 * math.pi


def wrap24(x):
    """Wrap hours to [0, 24)."""
    y = np.mod(x, 24.0)
    y = np.where(y >= 24.0 - 1e-9, 0.0, y)
    return float(y) if np.ndim(y) == 0 else y


def wrap12(x):
    """Wrap a signed hour difference to [-12, 12)."""
    return np.mod(np.asarray(x, dtype=float) + 12.0, 24.0) - 12.0


def circ_diff(a, b):
    """Signed circular difference a - b in hours, in [-12, 12)."""
    return wrap12(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def _to_angle(h):
    return np.asarray(h, dtype=float) * TWO_PI / 24.0


def circ_mean(hours: Iterable[float], weights: Optional[Iterable[float]] = None) -> float:
    """Circular mean of clock times (hours). NaNs are ignored."""
    h = np.asarray(list(hours), dtype=float)
    w = np.ones_like(h) if weights is None else np.asarray(list(weights), dtype=float)
    ok = ~np.isnan(h) & ~np.isnan(w)
    if not ok.any():
        return float("nan")
    a = _to_angle(h[ok])
    s = np.sum(w[ok] * np.sin(a))
    c = np.sum(w[ok] * np.cos(a))
    if math.hypot(s, c) < 1e-12:
        return float("nan")
    return float(wrap24(math.atan2(s, c) * 24.0 / TWO_PI))


def circ_resultant(hours: Iterable[float]) -> float:
    h = np.asarray(list(hours), dtype=float)
    h = h[~np.isnan(h)]
    if h.size == 0:
        return float("nan")
    a = _to_angle(h)
    return float(math.hypot(np.mean(np.sin(a)), np.mean(np.cos(a))))


def circ_sd(hours: Iterable[float]) -> float:
    """Circular standard deviation in hours: sqrt(-2 ln R) * 24 / 2pi."""
    r = circ_resultant(hours)
    if not np.isfinite(r):
        return float("nan")
    r = min(max(r, 1e-12), 1.0)
    return float(math.sqrt(-2.0 * math.log(r)) * 24.0 / TWO_PI)


def parse_clock(value) -> float:
    """'21:10' -> 21.1667; floats are returned wrapped to [0, 24)."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.floating)):
        return float(wrap24(value))
    s = str(value).strip()
    parts = s.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Cannot parse clock time {value!r}; expected 'HH:MM'")
    h, m = int(parts[0]), int(parts[1])
    sec = float(parts[2]) if len(parts) == 3 else 0.0
    if not (0 <= h <= 24 and 0 <= m < 60):
        raise ValueError(f"Invalid clock time {value!r}")
    return float(wrap24(h + m / 60.0 + sec / 3600.0))


def fmt_clock(h: float) -> str:
    if h is None or not np.isfinite(h):
        return "NA"
    total = int(round(float(wrap24(h)) * 60.0)) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


def fmt_signed(h: float, digits: int = 1) -> str:
    if h is None or not np.isfinite(h):
        return "NA"
    return f"{h:+.{digits}f} h"


def to_local(values, tz: str) -> pd.Series:
    """Parse to timezone-aware datetimes in `tz`.

    Naive values are interpreted as local civil time in `tz`; ambiguous or
    non-existent local times (DST transitions) raise, because silently
    guessing would introduce a 1-h error.
    """
    raw = pd.Series(list(values) if not isinstance(values, pd.Series) else values.to_list())
    try:
        s = pd.to_datetime(raw, utc=False)
    except (ValueError, TypeError):
        # mixed UTC offsets (e.g. across a DST transition): all values must carry an offset
        if any(getattr(pd.Timestamp(v), "tzinfo", None) is None for v in raw):
            raise ValueError("mix of timezone-aware and naive timestamps; make them consistent")
        s = pd.to_datetime(raw, utc=True)
    if s.dtype == object:
        s = pd.to_datetime(raw, utc=True)
    if getattr(s.dt, "tz", None) is None:
        try:
            s = s.dt.tz_localize(tz, ambiguous="raise", nonexistent="raise")
        except Exception as exc:  # pragma: no cover - message path
            raise ValueError(
                "Naive timestamps fall on a DST transition in "
                f"{tz}; supply timezone-aware timestamps. ({exc})"
            ) from exc
    else:
        s = s.dt.tz_convert(tz)
    return s.reset_index(drop=True)


def to_local_scalar(value, tz: str) -> pd.Timestamp:
    return to_local([value], tz).iloc[0]


def clock_hours(ts: pd.Series) -> np.ndarray:
    """Local civil clock time in hours for tz-aware datetimes."""
    ts = pd.Series(ts)
    return (
        ts.dt.hour.to_numpy(float)
        + ts.dt.minute.to_numpy(float) / 60.0
        + ts.dt.second.to_numpy(float) / 3600.0
        + ts.dt.microsecond.to_numpy(float) / 3.6e9
    )


def utc_ns(ts) -> np.ndarray:
    """Nanoseconds since epoch (UTC) for tz-aware datetimes."""
    s = pd.Series(ts)
    if len(s) == 0:
        return np.array([], dtype=np.int64)
    idx = pd.DatetimeIndex(s.dt.tz_convert("UTC")).as_unit("ns")
    return idx.asi8.astype(np.int64)


def local_midnight(date, tz: str) -> pd.Timestamp:
    """Local midnight of a calendar date as a tz-aware Timestamp."""
    return pd.Timestamp(pd.Timestamp(date).date()).tz_localize(tz)


def hours_after_local_midnight(ts: pd.Timestamp, date, tz: str) -> float:
    """Elapsed (UTC) hours from local midnight of `date` to `ts`."""
    return (ts - local_midnight(date, tz)).total_seconds() / 3600.0


def instant_from_date_hours(date, hours: float, tz: str) -> pd.Timestamp:
    """Inverse of hours_after_local_midnight."""
    return (local_midnight(date, tz) + pd.Timedelta(hours=float(hours))).tz_convert(tz)


def evening_date(ts: pd.Timestamp):
    """Calendar date of the evening an event belongs to (shift by -12 h).

    A DLMO at 00:30 belongs to the previous evening; one at 19:00 to the same day.
    """
    return (ts - pd.Timedelta(hours=12)).date()


def unwrap_sequence(hours: Iterable[float], start_ref: Optional[float] = None) -> np.ndarray:
    """Unwrap a daily sequence of clock times so consecutive steps are in [-12, 12)."""
    h = np.asarray(list(hours), dtype=float)
    out = np.full_like(h, np.nan)
    prev = start_ref
    for i, v in enumerate(h):
        if np.isnan(v):
            continue
        if prev is None or np.isnan(prev):
            out[i] = v
        else:
            out[i] = prev + float(wrap12(v - prev))
        prev = out[i]
    return out
