"""Synthetic participants with a known true DLMO trajectory (for demos and software validation).

The generative model is deliberately simple and shares structure with the state-space
model (sleep follows DLMO with a phase angle), so accuracy figures obtained on it are
optimistic and say nothing about real-world validity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ._time import instant_from_date_hours
from .phase import PhaseObservation


@dataclass
class SimConfig:
    n_days: int = 21
    start: str = "2026-03-02"          # a Monday
    tz: str = "Europe/Berlin"
    base_mean_h: float = 21.0
    base_sd_h: float = 1.0
    free_delay_slope: float = 0.3      # later chronotypes shift more on free days
    free_delay_intercept_h: float = 0.4
    adapt_rate: float = 0.5            # fraction of the gap to the target closed per day
    daily_noise_h: float = 0.1
    onset_angle_mean_h: float = 2.0
    onset_angle_between_sd_h: float = 0.5
    onset_night_sd_h: float = 0.4
    free_evening_behavioural_delay_h: float = 0.5
    sleep_need_mean_h: float = 8.0
    sleep_need_night_sd_h: float = 0.5
    alarm_mean_h: float = 6.5
    alarm_between_sd_h: float = 0.4
    shift_prob: float = 0.0            # probability of a sustained phase delay
    shift_size_h: float = 2.0
    shift_day: int = 12


def simulate_participant(seed: int, cfg: Optional[SimConfig] = None) -> dict:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(seed)
    tz = cfg.tz
    dates = [d.date() for d in pd.date_range(cfg.start, periods=cfg.n_days, freq="D")]
    base = rng.normal(cfg.base_mean_h, cfg.base_sd_h)
    delay = float(np.clip(cfg.free_delay_intercept_h + cfg.free_delay_slope * (base - 21.0), 0.0, 1.5))
    onset_angle = cfg.onset_angle_mean_h + rng.normal(0, cfg.onset_angle_between_sd_h)
    alarm = cfg.alarm_mean_h + rng.normal(0, cfg.alarm_between_sd_h)
    shifted = rng.random() < cfg.shift_prob

    phi = base
    true_rows, sleep_rows = [], []
    for i, d in enumerate(dates):
        next_is_work = pd.Timestamp(d + pd.Timedelta(days=1)).dayofweek < 5
        target = base + (0.0 if next_is_work else delay) + (cfg.shift_size_h if shifted and i >= cfg.shift_day else 0.0)
        phi = phi + (cfg.adapt_rate * (target - phi) if i > 0 else 0.0) + rng.normal(0, cfg.daily_noise_h)
        true_rows.append(dict(date=d, dlmo_h=phi, free_evening=not next_is_work, shifted=shifted))
        onset_h = phi + onset_angle + rng.normal(0, cfg.onset_night_sd_h) + (0 if next_is_work else cfg.free_evening_behavioural_delay_h)
        end_h = onset_h + cfg.sleep_need_mean_h + rng.normal(0, cfg.sleep_need_night_sd_h)
        woke_by_alarm = False
        if next_is_work and end_h > 24.0 + alarm:
            end_h, woke_by_alarm = 24.0 + alarm + abs(rng.normal(0, 0.05)), True
        sleep_rows.append(dict(sleep_onset=instant_from_date_hours(d, onset_h, tz),
                               sleep_end=instant_from_date_hours(d, end_h, tz),
                               alarm=woke_by_alarm, work_day=next_is_work))
    return dict(truth=pd.DataFrame(true_rows), sleep=pd.DataFrame(sleep_rows),
                meta=dict(base=base, free_delay=delay, onset_angle=onset_angle, shifted=shifted))


def simulate_observation(truth: pd.DataFrame, evening_index: int, source: str = "HairTime",
                         sample_hours_after_midnight_next_day: float = 10.0, error_sd_h: float = 1.46,
                         tz: str = "Europe/Berlin", rng: Optional[np.random.Generator] = None) -> PhaseObservation:
    """Create an assay (morning sample) or measured-DLMO (evening) observation with Gaussian error."""
    rng = rng or np.random.default_rng()
    row = truth.iloc[evening_index]
    d = row["date"]
    true_inst = instant_from_date_hours(d, row["dlmo_h"], tz)
    err = rng.normal(0, error_sd_h)
    if source == "DLMO":
        sampled = true_inst + pd.Timedelta(hours=1.0)
        est = true_inst + pd.Timedelta(hours=err)
        return PhaseObservation(source="DLMO", estimate_clock=f"{est.hour:02d}:{est.minute:02d}", sampled_at=sampled,
                                day_type="workday" if not row["free_evening"] else "free")
    sampled = instant_from_date_hours(d, 24.0 + sample_hours_after_midnight_next_day, tz)
    internal = (sampled - true_inst).total_seconds() / 3600 - err
    return PhaseObservation(source=source, internal_time_h=internal, sampled_at=sampled,
                            day_type="free" if pd.Timestamp(sampled).dayofweek >= 5 else "workday")
