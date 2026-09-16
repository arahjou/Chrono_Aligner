"""Phase observations (measured DLMO, assay-predicted DLMO, model-predicted DLMO).

The engine *consumes* assay outputs; it does not implement HairTime or BodyTime.

Time bookkeeping
----------------
An assay estimates the internal time of the sample relative to DLMO. The DLMO
instance it refers to is therefore

    dlmo_instant = sampled_at - internal_time_h

so an error in the recorded sampling time propagates one-to-one into phase.
Each DLMO instance is labelled with its *evening date* (date of instant - 12 h):
a HairTime sample taken on the morning of 4 March at 13 h after pDLMO refers to
the DLMO of the evening of 3 March.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ._time import (
    circ_diff, circ_mean, evening_date, fmt_clock, hours_after_local_midnight, parse_clock,
    to_local_scalar, wrap24,
)

# Source defaults. base_sd_h values are derived from published validation summaries and
# are *starting points*, not calibrated error models:
#   DLMO       ~0.5 h measurement error (cited by Wittenbrink et al., 2018)
#   HairTime   Bland-Altman LoA -2.84..+2.90 h -> SD ~1.46 h (Maier et al., 2025 preprint)
#   BodyTime   median absolute error 0.54-0.69 h -> SD ~0.8 h if roughly normal
#   light_model SD 1.1 h with 9 days of data (Woelders et al., 2017)
SOURCES: Dict[str, dict] = {
    "DLMO": dict(marker="DLMO", tier=0, base_sd_h=0.5, window=None,
                 validity="reference standard; protocol- and threshold-dependent"),
    "HairTime": dict(marker="pDLMO", tier=1, base_sd_h=1.46, window=(9.0, 17.0),
                     validity="healthy adults, regular schedule, daytime sample 9-17 h after DLMO"),
    "BodyTime": dict(marker="pDLMO", tier=1, base_sd_h=0.8, window=None,
                     validity="blood monocytes; constant-routine training, external validation in early/late chronotypes"),
    "light_model": dict(marker="pDLMO", tier=2, base_sd_h=1.1, window=None,
                        validity="Kronauer-type model; >= ~9 days of light data"),
}


@dataclass
class QCPolicy:
    """How assay quality indicators widen uncertainty.

    Defaults are deliberately conservative placeholders. They should be calibrated
    against validation data (e.g., the HairTime validation cohort) before use.
    """
    add_model_sd: bool = True              # sd = sqrt(sd^2 + model_sd_h^2)
    amplitude_threshold: Optional[float] = None  # set when a calibrated cut-off exists
    low_amplitude_factor: float = 1.25
    outside_window_factor: float = 1.5
    unvalidated_population_factor: float = 1.5


@dataclass
class PhaseObservation:
    source: str
    estimate_clock: Optional[object] = None      # "21:10" or hours
    sampled_at: Optional[object] = None          # ISO string / Timestamp (exact, with offset)
    internal_time_h: Optional[float] = None      # assay output: hours after DLMO at sampling
    marker: Optional[str] = None                 # "DLMO" | "pDLMO"
    day_type: Optional[str] = None               # "workday" | "free"
    uncertainty_h: Optional[float] = None        # explicit SD (overrides base SD)
    qc: Dict[str, object] = field(default_factory=dict)
    validity_domain: Optional[str] = None
    tz: Optional[str] = None

    # resolved fields
    dlmo_instant: Optional[pd.Timestamp] = field(default=None, init=False)
    dlmo_date: Optional[object] = field(default=None, init=False)
    dlmo_hours: float = field(default=float("nan"), init=False)   # hours after local midnight of dlmo_date
    hours_after_dlmo: float = field(default=float("nan"), init=False)
    in_validated_window: Optional[bool] = field(default=None, init=False)
    effective_sd_h: float = field(default=float("nan"), init=False)
    flags: List[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source!r}; known: {sorted(SOURCES)}")
        default_marker = SOURCES[self.source]["marker"]
        if self.marker is None:
            self.marker = default_marker
        if self.source != "DLMO" and self.marker == "DLMO":
            raise ValueError("assay/model outputs must use marker='pDLMO' (predicted is not measured)")
        if self.validity_domain is None:
            self.validity_domain = SOURCES[self.source]["validity"]

    # ------------------------------------------------------------------
    @property
    def label(self) -> str:
        return "DLMO" if self.marker == "DLMO" else f"pDLMO[{self.source}]"

    @property
    def tier(self) -> int:
        return SOURCES[self.source]["tier"]

    def resolve(self, tz: str, policy: Optional[QCPolicy] = None, schedule_type: str = "regular") -> "PhaseObservation":
        policy = policy or QCPolicy()
        tz = self.tz or tz
        self.flags = []
        if self.sampled_at is None:
            raise ValueError(f"{self.label}: sampled_at is required (the sampling timestamp is part of the measurement)")
        sampled = to_local_scalar(self.sampled_at, tz)
        if isinstance(self.sampled_at, str) and ("+" not in self.sampled_at[10:] and "Z" not in self.sampled_at):
            self.flags.append("sampled_at had no UTC offset; interpreted as local time in " + tz)

        if self.internal_time_h is not None:
            instant = sampled - pd.Timedelta(hours=float(self.internal_time_h))
            if self.estimate_clock is not None:
                est = parse_clock(self.estimate_clock)
                loc = instant.hour + instant.minute / 60 + instant.second / 3600
                if abs(float(circ_diff(est, loc))) > 0.1:
                    raise ValueError(f"{self.label}: estimate_clock and internal_time_h are inconsistent")
        elif self.estimate_clock is not None:
            est = parse_clock(self.estimate_clock)
            day0 = pd.Timestamp(sampled.date())
            candidates = [
                (day0 + pd.Timedelta(days=k)).tz_localize(tz, nonexistent="shift_forward", ambiguous=False)
                + pd.Timedelta(hours=est)
                for k in (-2, -1, 0, 1)
            ]
            if self.marker == "DLMO":
                # measured DLMO: the instance closest to the sampling (protocol) time
                instant = min(candidates, key=lambda c: abs((c - sampled).total_seconds()))
            else:
                # assay: most recent DLMO at or before the sample
                before = [c for c in candidates if c <= sampled]
                instant = max(before)
        else:
            raise ValueError(f"{self.label}: give estimate_clock or internal_time_h")

        self.dlmo_instant = instant.tz_convert(tz)
        self.dlmo_date = evening_date(self.dlmo_instant)
        self.dlmo_hours = hours_after_local_midnight(self.dlmo_instant, self.dlmo_date, tz)
        self.hours_after_dlmo = (sampled - self.dlmo_instant).total_seconds() / 3600.0
        if self.estimate_clock is None:
            i = self.dlmo_instant
            self.estimate_clock = fmt_clock(i.hour + i.minute / 60 + i.second / 3600)

        # uncertainty
        sd = float(self.uncertainty_h) if self.uncertainty_h is not None else SOURCES[self.source]["base_sd_h"]
        explicit = self.uncertainty_h is not None
        model_sd = self.qc.get("model_sd_h")
        if model_sd is not None and policy.add_model_sd and not explicit:
            sd = math.sqrt(sd ** 2 + float(model_sd) ** 2)
            self.flags.append(f"model disagreement {float(model_sd):.2f} h added to uncertainty")
        amp = self.qc.get("amplitude_score")
        if amp is not None and not self.qc.get("low_amplitude_adjusted", False):
            if policy.amplitude_threshold is None:
                self.flags.append("amplitude_score present but no calibrated threshold configured (QCPolicy.amplitude_threshold)")
            elif float(amp) < policy.amplitude_threshold and not explicit:
                sd *= policy.low_amplitude_factor
                self.flags.append("low circadian amplitude: uncertainty widened")

        window = SOURCES[self.source]["window"]
        if window is not None:
            lo, hi = window
            self.in_validated_window = bool(lo <= self.hours_after_dlmo <= hi)
            if "in_validated_window" in self.qc and bool(self.qc["in_validated_window"]) != self.in_validated_window:
                self.flags.append("qc.in_validated_window disagrees with the recomputed value; recomputed value used")
            if not self.in_validated_window:
                sd *= policy.outside_window_factor
                self.flags.append(
                    f"sample at {self.hours_after_dlmo:.1f} h after pDLMO is outside the validated window {lo:g}-{hi:g} h"
                )
        if self.tier == 1 and schedule_type == "shift":
            sd *= policy.unvalidated_population_factor
            self.flags.append("assay not validated against DLMO in shift work / jet lag: uncertainty widened")
        self.effective_sd_h = sd
        return self

    def to_row(self) -> dict:
        return dict(
            label=self.label, source=self.source, marker=self.marker, estimate=fmt_clock(wrap24(self.dlmo_hours)),
            dlmo_date=self.dlmo_date, dlmo_instant=self.dlmo_instant, sampled_at=self.sampled_at,
            hours_after_dlmo=round(self.hours_after_dlmo, 2), in_window=self.in_validated_window,
            day_type=self.day_type, sd_h=round(self.effective_sd_h, 2), flags="; ".join(self.flags),
        )


@dataclass
class CombinedPhase:
    hours: float                # clock hours [0,24) of the combined anchor
    sd_h: float
    label: str
    dates: List[object]
    n: int
    flags: List[str] = field(default_factory=list)


def combine(observations: Sequence[PhaseObservation], independent: bool = True) -> CombinedPhase:
    """Inverse-variance weighted circular mean of resolved observations.

    Only valid if their errors can be considered independent; otherwise the
    observation with the smallest SD is returned.
    """
    obs = list(observations)
    if not obs:
        raise ValueError("no observations to combine")
    labels = sorted({o.label for o in obs})
    label = labels[0] if len(labels) == 1 else "pDLMO[" + "+".join(sorted({o.source for o in obs})) + "]"
    dates = sorted({o.dlmo_date for o in obs})
    if len(obs) == 1 or not independent:
        best = min(obs, key=lambda o: o.effective_sd_h)
        return CombinedPhase(float(wrap24(best.dlmo_hours)), best.effective_sd_h, best.label, [best.dlmo_date], 1)
    w = np.array([1.0 / o.effective_sd_h ** 2 for o in obs])
    mean = circ_mean([o.dlmo_hours for o in obs], weights=w)
    sd = float(1.0 / math.sqrt(w.sum()))
    flags = []
    if len(dates) > 1:
        flags.append(f"pooled across {len(dates)} dates: assumes phase is constant between {dates[0]} and {dates[-1]}")
    return CombinedPhase(mean, sd, label, dates, len(obs), flags)


def disagreements(observations: Sequence[PhaseObservation], z: float = 1.96) -> List[dict]:
    """Pairs whose difference exceeds z * combined SD. Reported, not averaged away."""
    out = []
    for a, b in combinations(observations, 2):
        diff = float(circ_diff(a.dlmo_hours, b.dlmo_hours)) if a.dlmo_date == b.dlmo_date else \
            (a.dlmo_instant - b.dlmo_instant).total_seconds() / 3600.0 - 24.0 * (a.dlmo_date - b.dlmo_date).days
        comb = math.sqrt(a.effective_sd_h ** 2 + b.effective_sd_h ** 2)
        if abs(diff) > z * comb:
            causes = "low amplitude, wrong sampling timestamp, misalignment"
            if a.dlmo_date != b.dlmo_date:
                causes += ", or real phase change between sampling days"
            out.append(dict(a=a.label, a_date=a.dlmo_date, b=b.label, b_date=b.dlmo_date,
                            diff_h=round(diff, 2), combined_sd_h=round(comb, 2), possible_causes=causes))
    return out
