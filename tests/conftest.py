import pandas as pd
import pytest

TZ = "Europe/Berlin"


def make_sleep(start="2026-02-25", days=14, work_onset=23.5, work_end=30.75, free_onset=24.5, free_end=33.5,
               alarm_on_free=False, shift=None):
    rows = []
    s = pd.Timestamp(start)
    for i in range(days):
        d = s + pd.Timedelta(days=i)
        nxt = (d + pd.Timedelta(days=1)).dayofweek < 5
        on, end = (work_onset, work_end) if nxt else (free_onset, free_end)
        rows.append(dict(sleep_onset=(d + pd.Timedelta(hours=on)).tz_localize(TZ),
                         sleep_end=(d + pd.Timedelta(hours=end)).tz_localize(TZ),
                         alarm=nxt or alarm_on_free, work_day=nxt,
                         shift_type=(shift if nxt else "free") if shift else None))
    df = pd.DataFrame(rows)
    if not shift:
        df = df.drop(columns="shift_type")
    return df


@pytest.fixture
def sleep():
    return make_sleep()


@pytest.fixture
def hair_obs():
    from chronoalign import PhaseObservation
    return PhaseObservation(source="HairTime", marker="pDLMO", estimate_clock="21:10",
                            sampled_at="2026-03-04T10:15:00+01:00", day_type="workday",
                            qc={"model_sd_h": 0.35, "amplitude_score": 3.1})
