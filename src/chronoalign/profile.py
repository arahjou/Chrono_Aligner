"""CircadianProfile: anchors + provenance + uncertainty + phase trajectory."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ._time import fmt_clock, instant_from_date_hours, wrap24
from .grades import GRADES
from .phase import PhaseObservation
from .sleep import SleepPhenotype

ROUTE_CAPABILITIES = {
    "A": ("CRT, hours since wake, prior sleep duration, sleep-phase time, wake-anchored scheduling",
          "physiological phase; phase angle of entrainment"),
    "B": ("DLMO-relative time; phase-anchored scheduling",
          "hours since wake; prior sleep; day-to-day change in phase; phase angle"),
    "C": ("all coordinates; phase angle; propagation of phase to days without a sample",
          "nothing structurally; accuracy limited by the validity of each source"),
}


@dataclass
class CircadianProfile:
    timezone: str
    route: str
    evidence_grade: str
    # physiological anchor
    phase_anchor: Optional[str] = None
    phase_anchor_h: float = float("nan")
    phase_anchor_type: Optional[str] = None
    phase_anchor_date: Optional[str] = None
    phase_uncertainty_h: float = float("nan")
    stability_assumption: Optional[str] = None
    phase_mode: str = "none"                  # none | fixed | filter | smoother
    trajectory: Optional[pd.DataFrame] = None # per evening date: dlmo_h, sd_h
    # behavioural anchor
    behavioural_anchor: Optional[str] = None
    behavioural_anchor_h: float = float("nan")
    behavioural_se_min: float = float("nan")
    # phase angle on the anchor day (sleep episode following the DLMO)
    phase_angle_onset_h: float = float("nan")
    phase_angle_midsleep_h: float = float("nan")
    phase_angle_wake_h: float = float("nan")
    # sleep summary
    n_days_sleep: int = 0
    n_free_days: int = 0
    social_jetlag_h: float = float("nan")
    sri: float = float("nan")
    schedule_type: str = "regular"
    sleep: Optional[SleepPhenotype] = None
    sleep_table: Optional[pd.DataFrame] = None
    observations: List[PhaseObservation] = field(default_factory=list)
    light_dlmo: Optional[pd.DataFrame] = None
    disagreements: List[dict] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def has_phase(self) -> bool:
        return self.phase_anchor_type is not None

    @property
    def has_sleep(self) -> bool:
        return self.sleep_table is not None

    @property
    def phase_is_measured(self) -> bool:
        return self.phase_anchor_type == "DLMO"

    def dlmo_for_date(self, date) -> tuple[float, float]:
        """(hours after local midnight, SD) of the DLMO on evening `date`."""
        if not self.has_phase:
            raise ValueError(self._missing("physiological phase"))
        d = pd.Timestamp(date).date()
        if self.trajectory is not None and self.phase_mode in ("filter", "smoother"):
            tr = self.trajectory
            col, sdcol = ("dlmo_h", "sd_h") if self.phase_mode == "smoother" else ("dlmo_filter_h", "sd_filter_h")
            hit = tr[tr["date"] == d]
            if len(hit):
                return float(hit[col].iloc[0]), float(hit[sdcol].iloc[0])
            # outside the fused range: hold the nearest value, flag via SD growth
            nearest = tr.iloc[(pd.to_datetime(tr["date"]) - pd.Timestamp(d)).abs().argmin()]
            gap = abs((pd.Timestamp(nearest["date"]) - pd.Timestamp(d)).days)
            return float(nearest[col]), float(np.hypot(nearest[sdcol], 0.25 * gap))
        h = self.phase_anchor_h
        return (h if h >= 12 else h + 24.0), self.phase_uncertainty_h

    def dlmo_instants(self, start_date, end_date) -> pd.DataFrame:
        dates = [d.date() for d in pd.date_range(start_date, end_date, freq="D")]
        rows = []
        for d in dates:
            h, sd = self.dlmo_for_date(d)
            rows.append(dict(date=d, dlmo_h=h, sd_h=sd, instant=instant_from_date_hours(d, h, self.timezone)))
        return pd.DataFrame(rows)

    def _missing(self, what: str) -> str:
        can, cannot = ROUTE_CAPABILITIES.get(self.route, ("", ""))
        return (f"{what} is not available for Route {self.route} (grade {self.evidence_grade}). "
                f"Available: {can}. Not available: {cannot}.")

    def summary(self) -> str:
        g = GRADES.get(self.evidence_grade, ("", "", ""))
        lines = [
            f"CircadianProfile  tz={self.timezone}  route={self.route}  grade={self.evidence_grade} ({g[0]})",
        ]
        if self.has_phase:
            lines.append(
                f"  phase anchor     {self.phase_anchor} {self.phase_anchor_type}  date={self.phase_anchor_date}  "
                f"SD={self.phase_uncertainty_h:.2f} h  mode={self.phase_mode}"
            )
            if self.stability_assumption:
                lines.append(f"  stability        {self.stability_assumption}")
        if self.behavioural_anchor:
            lines.append(f"  behavioural      MSFsc {self.behavioural_anchor} (SE {abs(self.behavioural_se_min):.0f} min)")
        if np.isfinite(self.phase_angle_onset_h):
            lines.append(
                f"  phase angle      onset {self.phase_angle_onset_h:+.2f} h, midsleep {self.phase_angle_midsleep_h:+.2f} h, "
                f"wake {self.phase_angle_wake_h:+.2f} h (relative to DLMO on the anchor evening)"
            )
        if self.has_sleep:
            lines.append(
                f"  sleep            n={self.n_days_sleep} free={self.n_free_days} SJL={self.social_jetlag_h:.2f} h "
                f"SRI={self.sri:.0f} schedule={self.schedule_type}"
            )
        for o in self.observations:
            lines.append(f"  observation      {o.label} {o.estimate_clock} (evening {o.dlmo_date}) SD={o.effective_sd_h:.2f} h"
                         + (f"  [{'; '.join(o.flags)}]" if o.flags else ""))
        for d in self.disagreements:
            lines.append(f"  DISAGREEMENT     {d['a']} ({d['a_date']}) vs {d['b']} ({d['b_date']}): {d['diff_h']:+.2f} h")
        for f in self.flags:
            lines.append(f"  note             {f}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()
