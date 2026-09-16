import math

import pandas as pd
import pytest

import chronoalign as chrono
from conftest import TZ, make_sleep


def test_doc_phase_observation_example(hair_obs):
    o = hair_obs.resolve(TZ)
    assert o.hours_after_dlmo == pytest.approx(13.083, abs=0.01)   # spec: 13.1
    assert o.in_validated_window is True
    assert str(o.dlmo_date) == "2026-03-03"                        # morning sample -> previous evening
    assert o.label == "pDLMO[HairTime]"


def test_internal_time_input_and_timestamp_error_propagates():
    a = chrono.PhaseObservation(source="HairTime", internal_time_h=13.0, sampled_at="2026-03-04T10:15:00+01:00").resolve(TZ)
    b = chrono.PhaseObservation(source="HairTime", internal_time_h=13.0, sampled_at="2026-03-04T10:45:00+01:00").resolve(TZ)
    assert (b.dlmo_instant - a.dlmo_instant).total_seconds() / 3600 == pytest.approx(0.5)


def test_predicted_cannot_be_labelled_measured():
    with pytest.raises(ValueError):
        chrono.PhaseObservation(source="HairTime", marker="DLMO", estimate_clock="21:00", sampled_at="2026-03-04T10:00:00+01:00")


def test_outside_window_widens_uncertainty():
    inside = chrono.PhaseObservation(source="HairTime", estimate_clock="21:00", sampled_at="2026-03-04T10:00:00+01:00").resolve(TZ)
    evening = chrono.PhaseObservation(source="HairTime", estimate_clock="21:00", sampled_at="2026-03-04T19:00:00+01:00").resolve(TZ)
    assert evening.in_validated_window is False
    assert evening.effective_sd_h == pytest.approx(inside.effective_sd_h * 1.5)


def test_disagreement_is_reported_not_averaged():
    a = chrono.PhaseObservation(source="HairTime", estimate_clock="20:00", sampled_at="2026-03-04T10:00:00+01:00").resolve(TZ)
    b = chrono.PhaseObservation(source="BodyTime", estimate_clock="23:59", sampled_at="2026-03-04T11:00:00+01:00").resolve(TZ)
    d = chrono.disagreements([a, b])
    assert len(d) == 1 and d[0]["diff_h"] == pytest.approx(-3.98, abs=0.02)


def test_measured_dlmo_preferred_over_assay(sleep, hair_obs):
    dlmo = chrono.PhaseObservation(source="DLMO", estimate_clock="21:40", sampled_at="2026-03-05T22:00:00+01:00")
    prof = chrono.fit(sleep=sleep, phase_observations=[hair_obs, dlmo], tz=TZ)
    assert prof.phase_anchor_type == "DLMO" and prof.phase_anchor == "21:40"


def test_routes_and_grades(sleep, hair_obs):
    assert chrono.fit(sleep=sleep, tz=TZ).route == "A"
    assert chrono.fit(sleep=sleep, tz=TZ).evidence_grade == "S1"
    pb = chrono.fit(phase_observations=[hair_obs], tz=TZ)
    assert (pb.route, pb.evidence_grade) == ("B", "P1")
    pc = chrono.fit(sleep=sleep, phase_observations=[hair_obs], tz=TZ)
    assert (pc.route, pc.evidence_grade) == ("C", "C")
    assert "phase constant over" in pc.stability_assumption


def test_route_a_cannot_give_phase(sleep):
    prof = chrono.fit(sleep=sleep, tz=TZ)
    df = pd.DataFrame({"datetime": pd.date_range("2026-03-02 08:00", periods=3, freq="h", tz=TZ)})
    with pytest.raises(ValueError, match="not available for Route A"):
        chrono.transform(df, prof, reference=["phase"])


def test_route_b_cannot_give_wake(hair_obs):
    prof = chrono.fit(phase_observations=[hair_obs], tz=TZ)
    df = pd.DataFrame({"datetime": pd.date_range("2026-03-04 08:00", periods=3, freq="h", tz=TZ)})
    with pytest.raises(ValueError, match="Route B"):
        chrono.transform(df, prof, reference=["wake"])


def test_phase_angle_on_sampling_evening(sleep, hair_obs):
    prof = chrono.fit(sleep=sleep, phase_observations=[hair_obs], tz=TZ)
    # evening 3 March (Tue, next day workday): onset 23:30 -> 23:30 - 21:10 = 2.33 h
    assert prof.phase_angle_onset_h == pytest.approx(2.333, abs=0.01)


def test_dynamic_filter_uncertainty_grows_away_from_sample(hair_obs):
    prof = chrono.fit(phase_observations=[hair_obs], tz=TZ, dynamic=True)
    h0, sd0 = prof.dlmo_for_date("2026-03-03")
    h5, sd5 = prof.dlmo_for_date("2026-03-08")
    assert h5 == pytest.approx(h0) and sd5 > sd0


def test_dynamic_does_not_turn_sleep_into_phase_level(sleep, hair_obs):
    prof = chrono.fit(sleep=sleep, phase_observations=[hair_obs], tz=TZ, dynamic=True)
    h, sd = prof.dlmo_for_date("2026-03-03")
    assert h == pytest.approx(21.1667, abs=0.1)      # level still set by the phase observation
    assert sd == pytest.approx(hair_obs.effective_sd_h, rel=0.1)
