import math

import numpy as np
import pandas as pd
import pytest

import chronoalign as chrono
from chronoalign._time import circ_mean, circ_sd, to_local, utc_ns, HOUR_NS, wrap12
from conftest import TZ, make_sleep


def test_circular_mean_across_midnight():
    assert circ_mean([23.5, 0.5]) == pytest.approx(0.0, abs=1e-9)
    assert circ_sd([3.0, 3.0, 3.0]) == pytest.approx(0.0, abs=1e-6)
    assert wrap12(23.0) == pytest.approx(-1.0)


def test_dst_durations_use_utc():
    s = to_local(["2026-03-28T23:00:00+01:00", "2026-03-29T07:00:00+02:00"], TZ)
    ns = utc_ns(s)
    assert (ns[1] - ns[0]) / HOUR_NS == pytest.approx(7.0)


def test_msfsc_correction_rule(sleep):
    ph, _ = chrono.sleep_phenotype(sleep, TZ)
    # work nights: 23:30-06:45 (7.25 h, mid 03:07.5); free: 00:30-09:30 (9 h, mid 05:00)
    assert ph.sd_w == pytest.approx(7.25)
    assert ph.sd_f == pytest.approx(9.0)
    sd_week = (7.25 * ph.n_work + 9.0 * ph.n_free) / (ph.n_work + ph.n_free)
    assert ph.sd_week == pytest.approx(sd_week)
    assert ph.msf == pytest.approx(5.0)
    assert ph.msfsc == pytest.approx(5.0 - (9.0 - sd_week) / 2)
    assert ph.social_jetlag_h == pytest.approx(5.0 - 3.125)


def test_msfsc_equals_msf_when_free_sleep_not_longer():
    s = make_sleep(free_onset=24.5, free_end=31.0)  # 6.5 h on free days < 7.25 h
    ph, _ = chrono.sleep_phenotype(s, TZ)
    assert ph.msfsc == pytest.approx(ph.msf)


def test_msfsc_not_computable_with_alarm_on_all_free_days():
    s = make_sleep(alarm_on_free=True)
    ph, _ = chrono.sleep_phenotype(s, TZ)
    assert math.isnan(ph.msfsc)
    assert "not computable" in ph.msfsc_note
    assert ph.n_free_alarm == 4


def test_s0_without_work_info(sleep):
    ph, _ = chrono.sleep_phenotype(sleep[["sleep_onset", "sleep_end"]], TZ)
    assert not ph.has_work_info and math.isnan(ph.msf)
    prof = chrono.fit(sleep=sleep[["sleep_onset", "sleep_end"]], tz=TZ)
    assert prof.evidence_grade == "S0"


def test_sri_perfectly_regular_is_100():
    rows = [dict(sleep_onset=pd.Timestamp("2026-01-05 23:00", tz=TZ) + pd.Timedelta(days=i),
                 sleep_end=pd.Timestamp("2026-01-06 07:00", tz=TZ) + pd.Timedelta(days=i)) for i in range(10)]
    assert chrono.sleep_regularity_index(chrono.sleep.prepare_sleep(pd.DataFrame(rows), TZ)) == pytest.approx(100.0)


def test_shift_work_no_trait_anchor():
    s = make_sleep(shift="night")
    ph, _ = chrono.sleep_phenotype(s, TZ)
    assert ph.schedule_type == "shift"
    assert math.isnan(ph.msfsc)
    assert "after_night" in ph.msf_by_shift
