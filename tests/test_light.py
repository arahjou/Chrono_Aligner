import numpy as np
import pandas as pd
import pytest

import chronoalign as chrono

TZ = "Europe/Berlin"


def _ld(start_hour, days=14):
    idx = pd.date_range("2026-03-02", periods=days * 240, freq="6min", tz=TZ)
    h = idx.hour + idx.minute / 60
    return pd.DataFrame({"datetime": idx, "lux": np.where((h >= start_hour) & (h < start_hour + 16), 500.0, 0.0)})


def test_light_model_entrains_and_tracks_schedule():
    a = chrono.light_model_dlmo(_ld(7), TZ)["dlmo_h"].tail(3).mean()
    b = chrono.light_model_dlmo(_ld(8), TZ)["dlmo_h"].tail(3).mean()
    assert 19.5 < a < 23.0
    assert b - a == pytest.approx(1.0, abs=0.1)


def test_matches_reference_implementation():
    circadian = pytest.importorskip("circadian")
    from circadian.models import Forger99
    t = np.arange(0, 10 * 24, 0.1)
    L = np.where(((t % 24) >= 7) & ((t % 24) < 23), 500.0, 0.0)
    ref = Forger99()
    ic = ref.equilibrate(t[:240], L[:240], num_loops=20)
    traj = ref(t, ic, L)
    mine = chrono.light.integrate_forger99(L, 0.1, ic)
    ref_dlmo = np.mod(ref.dlmos()[-3:], 24)
    my_min = chrono.light._minima(t, mine[:, 0])[-3:] - 7.0
    assert np.allclose(np.mod(my_min, 24), ref_dlmo, atol=0.15)


def test_grade_m_and_fusion_with_light():
    prof = chrono.fit(light=_ld(7), tz=TZ, dynamic=True)
    assert prof.evidence_grade == "M" and prof.phase_anchor_type == "pDLMO[light_model]"
    assert prof.trajectory is not None
