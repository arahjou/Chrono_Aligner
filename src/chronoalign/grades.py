"""Evidence grades (type of evidence, not guaranteed individual accuracy)."""
from __future__ import annotations

GRADES = {
    "S0": ("Sleep onset and end only", "Mean midsleep, regularity", "Descriptive only (free days cannot be identified)"),
    "S1": ("Sleep + alarm/work/free-day (shift type)", "MSFsc, social jetlag", "Behavioural alignment in regular schedules"),
    "M": ("Light and activity (>= about 9 days)", "pDLMO[model]", "Phase estimation, particularly under misalignment"),
    "P1": ("Single-sample assay with QC", "pDLMO[HairTime / BodyTime]", "Physiological anchor within validated domains"),
    "P2": ("Measured DLMO", "DLMO", "Reference, calibration and validation"),
    "C": ("Phase observation(s) + sleep (+/- light/activity)", "DLMO/pDLMO trajectory, MSFsc, phase angle",
          "Full coordinate set; within-person dynamics"),
}


def assign_grade(has_sleep: bool, has_work_info: bool, has_light: bool, sources: set) -> str:
    """Assign the evidence grade from the inputs that were actually used."""
    phase_sources = sources - {"light_model"}
    if phase_sources and has_sleep:
        return "C"
    if "DLMO" in phase_sources:
        return "P2"
    if phase_sources:
        return "P1"
    if has_light or "light_model" in sources:
        return "M"
    if has_sleep and has_work_info:
        return "S1"
    if has_sleep:
        return "S0"
    raise ValueError("no usable input")


def route(has_sleep: bool, has_phase: bool) -> str:
    if has_sleep and has_phase:
        return "C"
    if has_phase:
        return "B"
    if has_sleep:
        return "A"
    return "-"
