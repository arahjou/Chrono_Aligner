"""Sleep phenotype: MCTQ-style metrics from diary/actigraphy episodes (Routes A and C).

Input: one row per *main* sleep episode with columns

    sleep_onset   timezone-aware datetime (sleep onset, not bedtime)
    sleep_end     timezone-aware datetime (final wake, not get-up time)
    alarm         True if woken by alarm or obligation            (optional*)
    work_day      True if the day FOLLOWING this sleep is a workday (optional*)
    shift_type    optional: morning / evening / night / free / ...

(*) Without `alarm` and `work_day` only evidence grade S0 metrics are available
(mean midsleep, duration, regularity): MSF, MSFsc and social jetlag need to know
which days are free.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ._time import (
    HOUR_NS, circ_diff, circ_mean, circ_sd, clock_hours, fmt_clock, to_local, utc_ns, wrap24,
)

SHIFT_TYPES_IRREGULAR = {"evening", "night", "late", "rotating"}


def prepare_sleep(sleep: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Validate and enrich a sleep-episode table. Returns a new, sorted DataFrame."""
    if sleep is None or len(sleep) == 0:
        raise ValueError("sleep table is empty")
    missing = {"sleep_onset", "sleep_end"} - set(sleep.columns)
    if missing:
        raise ValueError(f"sleep table is missing columns: {sorted(missing)}")
    df = sleep.copy()
    df = df.reset_index(drop=True)
    df["onset"] = to_local(df["sleep_onset"], tz)
    df["end"] = to_local(df["sleep_end"], tz)
    df = df.sort_values("onset").reset_index(drop=True)

    on_ns, end_ns = utc_ns(df["onset"]), utc_ns(df["end"])
    dur = (end_ns - on_ns) / HOUR_NS
    if np.any(dur <= 0):
        bad = df.index[dur <= 0].tolist()
        raise ValueError(f"sleep_end must be after sleep_onset (rows {bad})")
    if np.any(dur > 20):
        warnings.warn("Some sleep episodes are longer than 20 h; check the input.")
    if len(df) > 1 and np.any(on_ns[1:] < end_ns[:-1]):
        warnings.warn("Overlapping sleep episodes detected; results may be unreliable.")

    df["duration_h"] = dur
    df["midsleep"] = df["onset"] + pd.to_timedelta(dur / 2.0, unit="h")
    df["midsleep"] = df["midsleep"].dt.tz_convert(tz)
    df["onset_clock_h"] = clock_hours(df["onset"])
    df["end_clock_h"] = clock_hours(df["end"])
    df["midsleep_clock_h"] = clock_hours(df["midsleep"])
    # The "night" is labelled by the evening it starts on (midsleep - 12 h).
    df["night_date"] = (df["midsleep"] - pd.Timedelta(hours=12)).dt.date
    df["wake_date"] = df["end"].dt.date

    has_alarm = "alarm" in df.columns and df["alarm"].notna().all()
    has_work = "work_day" in df.columns and df["work_day"].notna().all()
    df.attrs["has_work_info"] = bool(has_alarm and has_work)
    if has_alarm:
        df["alarm"] = df["alarm"].astype(bool)
    if has_work:
        df["work_day"] = df["work_day"].astype(bool)
    if "shift_type" in df.columns:
        df["shift_type"] = df["shift_type"].astype("string").str.lower()
    return df


def sleep_regularity_index(prep: pd.DataFrame, epoch_min: int = 1) -> float:
    """Sleep Regularity Index (Phillips et al., 2017), range -100..100.

    Probability of being in the same state (asleep/awake) at two time points
    24 h apart, rescaled. Only recorded main sleep episodes count as sleep, so
    naps that were not logged are treated as wake.
    """
    on = utc_ns(prep["onset"]) // (60 * 10**9 * epoch_min)
    end = utc_ns(prep["end"]) // (60 * 10**9 * epoch_min)
    start, stop = int(on.min()), int(end.max())
    n = stop - start
    lag = int(24 * 60 / epoch_min)
    if n <= lag:
        return float("nan")
    state = np.zeros(n, dtype=np.int8)
    for a, b in zip(on - start, end - start):
        state[int(a):int(b)] = 1
    same = state[lag:] == state[:-lag]
    return float(-100.0 + 200.0 * same.mean())


@dataclass
class SleepPhenotype:
    n_days: int
    has_work_info: bool
    n_work: int = 0
    n_free: int = 0
    n_free_alarm: int = 0
    mean_midsleep: float = float("nan")
    mean_duration_h: float = float("nan")
    sd_w: float = float("nan")
    sd_f: float = float("nan")
    sd_week: float = float("nan")
    so_w: float = float("nan")
    so_f: float = float("nan")
    se_w: float = float("nan")
    se_f: float = float("nan")
    msw: float = float("nan")
    msf: float = float("nan")
    msfsc: float = float("nan")
    msfsc_se_min: float = float("nan")
    msfsc_note: str = ""
    social_jetlag_h: float = float("nan")
    social_jetlag_sc_h: float = float("nan")
    weekly_sleep_loss_h: float = float("nan")
    midsleep_csd_h: float = float("nan")
    onset_csd_h: float = float("nan")
    wake_csd_h: float = float("nan")
    sri: float = float("nan")
    schedule_type: str = "regular"
    msf_by_shift: Dict[str, float] = field(default_factory=dict)
    typical: Dict[str, Dict[str, float]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"days={self.n_days} work={self.n_work} free(alarm-free)={self.n_free} free_with_alarm={self.n_free_alarm}",
            f"MSW={fmt_clock(self.msw)} MSF={fmt_clock(self.msf)} MSFsc={fmt_clock(self.msfsc)}"
            f" (SE {abs(self.msfsc_se_min):.0f} min) {self.msfsc_note}".rstrip(),
            f"SJL={self.social_jetlag_h:.2f} h  SJLsc={self.social_jetlag_sc_h:.2f} h  "
            f"weekly sleep loss={self.weekly_sleep_loss_h:.2f} h",
            f"circular SD onset/mid/wake = {self.onset_csd_h:.2f}/{self.midsleep_csd_h:.2f}/{self.wake_csd_h:.2f} h  SRI={self.sri:.0f}",
            f"schedule_type={self.schedule_type}",
        ]
        return "\n".join(lines)


def _mean(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)) if x.size else float("nan")


def sleep_phenotype(
    sleep: pd.DataFrame,
    tz: str,
    min_days: int = 7,
    min_free_days: int = 2,
    preferred_days: int = 14,
    schedule_type: Optional[str] = None,
) -> tuple[SleepPhenotype, pd.DataFrame]:
    """Compute the sleep phenotype. Returns (phenotype, prepared episode table)."""
    prep = prepare_sleep(sleep, tz)
    n = len(prep)
    ph = SleepPhenotype(n_days=n, has_work_info=prep.attrs["has_work_info"])
    if n < min_days:
        ph.warnings.append(f"only {n} sleep episodes (< minimum {min_days})")
    elif n < preferred_days:
        ph.warnings.append(f"{n} sleep episodes; {preferred_days} are preferred")

    ph.mean_midsleep = circ_mean(prep["midsleep_clock_h"])
    ph.mean_duration_h = _mean(prep["duration_h"])
    ph.midsleep_csd_h = circ_sd(prep["midsleep_clock_h"])
    ph.onset_csd_h = circ_sd(prep["onset_clock_h"])
    ph.wake_csd_h = circ_sd(prep["end_clock_h"])
    ph.sri = sleep_regularity_index(prep)

    # schedule type
    if schedule_type is not None:
        ph.schedule_type = schedule_type
    elif "shift_type" in prep.columns and prep["shift_type"].isin(SHIFT_TYPES_IRREGULAR).any():
        ph.schedule_type = "shift"

    if not ph.has_work_info:
        ph.msfsc_note = "no alarm/work_day information: free days cannot be identified (grade S0)"
        return ph, prep

    work = prep[prep["work_day"]]
    free_all = prep[~prep["work_day"]]
    free = free_all[~free_all["alarm"]]
    ph.n_work, ph.n_free, ph.n_free_alarm = len(work), len(free), int(free_all["alarm"].sum())
    if ph.n_free_alarm:
        ph.warnings.append(f"{ph.n_free_alarm} free day(s) with alarm excluded from MSF")

    ph.sd_w, ph.sd_f = _mean(work["duration_h"]), _mean(free["duration_h"])
    ph.so_w, ph.so_f = circ_mean(work["onset_clock_h"]), circ_mean(free["onset_clock_h"])
    ph.se_w, ph.se_f = circ_mean(work["end_clock_h"]), circ_mean(free["end_clock_h"])
    ph.msw, ph.msf = circ_mean(work["midsleep_clock_h"]), circ_mean(free["midsleep_clock_h"])

    # Typical clock times for the predicted-mode scheduler.
    #   wake on day d  -> grouped by the type of day d (work_day flag of the night ending on d)
    #   onset on evening d -> grouped by the type of the FOLLOWING day (same flag, night starting on d)
    is_work = prep["work_day"].to_numpy()
    for label, mask in (("workday", is_work), ("free", ~is_work)):
        ph.typical[label] = {
            "wake": circ_mean(prep.loc[mask, "end_clock_h"]),
            "onset_before": circ_mean(prep.loc[mask, "onset_clock_h"]),
        }

    if len(work) and len(free):
        ph.sd_week = (ph.sd_w * ph.n_work + ph.sd_f * ph.n_free) / (ph.n_work + ph.n_free)
        ph.social_jetlag_h = abs(float(circ_diff(ph.msf, ph.msw)))
        # Sleep-corrected social jetlag (Jankowski, 2017)
        if ph.sd_w <= ph.sd_f:
            ph.social_jetlag_sc_h = abs(float(circ_diff(ph.so_f, ph.so_w)))
        else:
            ph.social_jetlag_sc_h = abs(float(circ_diff(ph.se_f, ph.se_w)))
        # MCTQ weekly sleep loss
        if ph.sd_week > ph.sd_w:
            ph.weekly_sleep_loss_h = (ph.sd_week - ph.sd_w) * ph.n_work
        else:
            ph.weekly_sleep_loss_h = (ph.sd_week - ph.sd_f) * ph.n_free

    # MCTQShift-style free-day midsleep per preceding shift type
    if "shift_type" in prep.columns:
        last_shift = None
        groups: Dict[str, list] = {}
        for _, row in prep.iterrows():
            if row["work_day"]:
                last_shift = row["shift_type"] if pd.notna(row["shift_type"]) else last_shift
            elif not row["alarm"]:
                key = f"after_{last_shift}" if last_shift else "after_unknown"
                groups.setdefault(key, []).append(row["midsleep_clock_h"])
        ph.msf_by_shift = {k: circ_mean(v) for k, v in groups.items()}

    # MSFsc
    if ph.schedule_type == "shift":
        ph.msfsc_note = ("shift work: no single behavioural anchor is reported as a trait; "
                         "see msf_by_shift (MCTQShift)")
        return ph, prep
    if ph.n_free < min_free_days:
        ph.msfsc_note = f"not computable: {ph.n_free} alarm-free free day(s) (< {min_free_days})"
        return ph, prep
    if len(work) == 0:
        ph.msfsc = ph.msf
        ph.msfsc_note = "no workdays recorded: MSFsc = MSF"
    elif ph.sd_f <= ph.sd_w:
        ph.msfsc = ph.msf
    else:
        ph.msfsc = float(wrap24(ph.msf - (ph.sd_f - ph.sd_week) / 2.0))
    ph.msfsc_se_min = circ_sd(free["midsleep_clock_h"]) / math.sqrt(ph.n_free) * 60.0
    return ph, prep
