import numpy as np
import pandas as pd
import pytest

import chronoalign as chrono
from conftest import TZ, make_sleep


def _profile_with(msfsc_clock, pdlmo_clock, sleep):
    """Profile with a given behavioural and physiological anchor (for worked examples)."""
    obs = chrono.PhaseObservation(source="HairTime", estimate_clock=pdlmo_clock, sampled_at="2026-03-04T10:00:00+01:00")
    p = chrono.fit(sleep=sleep, phase_observations=[obs], tz=TZ)
    p.behavioural_anchor_h = chrono._time.parse_clock(msfsc_clock)
    return p


def test_doc_crt_vs_pdlmo_table(sleep):
    df = pd.DataFrame({"datetime": [pd.Timestamp("2026-03-05 08:00", tz=TZ)]})
    a = chrono.transform(df, _profile_with("04:00", "20:30", sleep), reference=["chronotype", "phase"])
    b = chrono.transform(df, _profile_with("04:00", "22:30", sleep), reference=["chronotype", "phase"])
    assert a["crt_phase_h"].iloc[0] == pytest.approx(4.0)
    assert a["pdlmo_phase_h"].iloc[0] == pytest.approx(11.5)
    assert b["pdlmo_phase_h"].iloc[0] == pytest.approx(9.5)
    assert a["anchor_source"].iloc[0] == "pDLMO[HairTime]"


def test_identity_linear_phase_equals_time_awake_plus_psi(sleep, hair_obs):
    prof = chrono.fit(sleep=sleep, phase_observations=[hair_obs], tz=TZ)
    t = pd.date_range("2026-03-04 07:00", "2026-03-04 21:00", freq="30min", tz=TZ)  # before the next DLMO instance
    out = chrono.transform(pd.DataFrame({"datetime": t}), prof, reference=["phase", "wake"])
    psi = out["pdlmo_phase_h"] - out["hours_since_wake"]
    assert psi.std() == pytest.approx(0.0, abs=1e-9)          # constant within one wake episode
    diag = chrono.psi_diagnostics(out)
    assert not diag["separation_supported"]


def test_hours_since_wake_across_dst():
    rows = [dict(sleep_onset=pd.Timestamp("2026-03-28T23:00:00+01:00"), sleep_end=pd.Timestamp("2026-03-29T07:00:00+02:00"),
                 alarm=False, work_day=False)]
    s = pd.concat([make_sleep(start="2026-03-15", days=13), pd.DataFrame(rows)], ignore_index=True)
    prof = chrono.fit(sleep=s, tz=TZ)
    out = chrono.transform(pd.DataFrame({"datetime": [pd.Timestamp("2026-03-29T09:00:00+02:00")]}), prof, reference=["wake"])
    assert out["hours_since_wake"].iloc[0] == pytest.approx(2.0)
    assert out["prior_sleep_duration_h"].iloc[0] == pytest.approx(7.0)   # 23:00 CET -> 07:00 CEST


def test_asleep_rows_are_nan(sleep):
    prof = chrono.fit(sleep=sleep, tz=TZ)
    out = chrono.transform(pd.DataFrame({"datetime": [pd.Timestamp("2026-03-03 02:00", tz=TZ)]}), prof, reference=["wake"])
    assert out["asleep"].iloc[0] and np.isnan(out["hours_since_wake"].iloc[0])


def test_timestamps_unchanged(sleep, hair_obs):
    prof = chrono.fit(sleep=sleep, phase_observations=[hair_obs], tz=TZ)
    df = pd.DataFrame({"datetime": pd.date_range("2026-03-02 08:00", periods=5, freq="h", tz=TZ), "glucose": [90, 95, 100, 98, 97]})
    out = chrono.transform(df, prof, reference=["phase", "chronotype", "wake"])
    pd.testing.assert_series_equal(out["datetime"], df["datetime"])


def test_wake_triggered_doc_example(sleep):
    prof = chrono.fit(sleep=sleep, tz=TZ)
    plan = chrono.schedule(prof, anchor="wake", offsets=[2, 7, 12], mode="wake_triggered",
                           wake_time="2026-03-04T07:15:00+01:00", predicted_onset="23:00")
    assert plan["clock"].tolist() == ["09:15", "14:15", "19:15"]


def test_late_waker_offset_after_sleep_flagged(sleep):
    prof = chrono.fit(sleep=sleep, tz=TZ)
    plan = chrono.schedule_wake(prof, [2, 8, 14], mode="wake_triggered",
                                wake_time="2026-03-04T10:00:00+01:00", predicted_onset="23:30")
    assert plan["clock"].tolist() == ["12:00", "18:00", "00:00"]
    assert plan["status"].tolist()[-1] == "after_sleep_limit"
    dropped = chrono.schedule_wake(prof, [2, 8, 14], mode="wake_triggered", rule="drop",
                                   wake_time="2026-03-04T10:00:00+01:00", predicted_onset="23:30")
    assert len(dropped) == 2


def test_constrained_doc_example():
    r = chrono.schedule_constrained("07:30", "21:00", events=3, first_after=1.5, last_before_sleep=2, min_spacing=4)
    assert r["times"] == ["09:00", "14:00", "19:00"] and r["valid"]
    assert r["valid_if_onset_at_or_after"] == "19:00"   # the constraint itself only needs 19:00


def test_predicted_mode_uses_day_type(sleep):
    prof = chrono.fit(sleep=sleep, tz=TZ)
    plan = chrono.schedule_wake(prof, [2], dates=["2026-03-04", "2026-03-07"], mode="predicted")
    assert plan["clock"].tolist() == ["08:45", "11:30"]   # workday wake 06:45, free-day wake 09:30


def test_phase_schedule_propagates_uncertainty(sleep, hair_obs):
    prof = chrono.fit(sleep=sleep, phase_observations=[hair_obs], tz=TZ)
    plan = chrono.schedule(prof, anchor="phase", offsets=[10, 16], dates=["2026-03-03"])
    assert plan["clock"].tolist() == ["07:10", "13:10"]
    assert plan["sd_h"].iloc[0] == pytest.approx(prof.phase_uncertainty_h, abs=0.01)


def test_sampling_window_doc_example():
    r = chrono.schedule_phase_sample(assay="HairTime", prior=None, prior_range=("19:00", "23:00"))
    assert r["robust_window"] == ("08:00", "12:00")
    assert r["recommended_time"] == "10:00"


def test_sampling_window_from_sleep_prior(sleep):
    prof = chrono.fit(sleep=sleep, tz=TZ)
    r = chrono.schedule_phase_sample(assay="HairTime", prior=prof)
    assert r["prior"].startswith("sleep-based") and r["robust_window"] is not None


def test_linear_reference_instance_default_and_explicit(sleep, hair_obs):
    prof = chrono.fit(sleep=sleep, phase_observations=[hair_obs], tz=TZ)
    df = pd.DataFrame({"datetime": [pd.Timestamp("2026-03-04 07:10", tz=TZ), pd.Timestamp("2026-03-05 07:10", tz=TZ)]})
    out = chrono.transform(df, prof, reference=["phase"])
    assert out["pdlmo_linear_h"].tolist() == pytest.approx([10.0, 34.0])     # DLMO 3 March 21:10 is k
    out2 = chrono.transform(df, prof, reference=["phase"], reference_date="2026-03-04")
    assert out2["pdlmo_linear_h"].tolist() == pytest.approx([-14.0, 10.0])
    assert out["pdlmo_phase_h"].tolist() == pytest.approx([10.0, 10.0])
