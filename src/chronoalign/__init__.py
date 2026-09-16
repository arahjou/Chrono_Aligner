"""ChronoAlign: a circadian reference engine.

    import chronoalign as chrono

    obs = chrono.PhaseObservation(source="HairTime", estimate_clock="21:10",
                                  sampled_at="2026-03-04T10:15:00+01:00", day_type="workday")
    profile = chrono.fit(sleep=sleep, phase_observations=[obs], tz="Europe/Berlin", dynamic=True)
    aligned = chrono.transform(cgm, profile, timestamp="datetime",
                               reference=["phase", "chronotype", "wake"])
    plan = chrono.schedule_phase_sample(assay="HairTime", prior=profile)
"""
from .api import fit
from .grades import GRADES, assign_grade
from .identifiability import concurvity_proxy, psi_diagnostics, psi_table, sensitivity_analysis
from .light import Forger99Params, light_model_dlmo
from .phase import SOURCES, PhaseObservation, QCPolicy, combine, disagreements
from .profile import CircadianProfile
from .schedule import schedule_constrained, schedule_phase, schedule_phase_sample, schedule_wake
from .sleep import SleepPhenotype, sleep_phenotype, sleep_regularity_index
from .statespace import StateSpaceConfig, fuse
from .transform import transform

__version__ = "0.2.0"


def schedule(profile, anchor="wake", offsets=(), **kwargs):
    """Dispatch: anchor='wake' -> schedule_wake, anchor='phase' -> schedule_phase."""
    if anchor == "wake":
        return schedule_wake(profile, offsets, **kwargs)
    if anchor == "phase":
        return schedule_phase(profile, offsets, **kwargs)
    raise ValueError("anchor must be 'wake' or 'phase'")


__all__ = [
    "fit", "transform", "schedule", "schedule_wake", "schedule_phase", "schedule_constrained",
    "schedule_phase_sample", "PhaseObservation", "QCPolicy", "CircadianProfile", "StateSpaceConfig",
    "SleepPhenotype", "sleep_phenotype", "sleep_regularity_index", "light_model_dlmo", "Forger99Params",
    "fuse", "combine", "disagreements", "psi_table", "psi_diagnostics", "concurvity_proxy",
    "sensitivity_analysis", "GRADES", "assign_grade", "SOURCES",
]
