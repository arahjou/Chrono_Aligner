"""End-to-end demo on one synthetic participant (Routes A, B and C)."""
import warnings

import numpy as np
import pandas as pd

import chronoalign as chrono
from chronoalign.synthetic import SimConfig, simulate_observation, simulate_participant

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 160)
TZ = "Europe/Berlin"

sim = simulate_participant(seed=7, cfg=SimConfig(n_days=14))
sleep, truth = sim["sleep"], sim["truth"]
print("True DLMO (first 3 evenings):", [chrono._time.fmt_clock(h) for h in truth["dlmo_h"][:3]])

# 1. Plan the hair sample from sleep records alone (Route A prior)
route_a = chrono.fit(sleep=sleep, tz=TZ)
print("\n--- Route A ---\n" + route_a.summary())
print("\nSampling plan:", chrono.schedule_phase_sample(assay="HairTime", prior=route_a))

# 2. A HairTime result arrives (sample on the morning after evening index 8)
obs = simulate_observation(truth, 8, rng=np.random.default_rng(1))
obs.qc = {"model_sd_h": 0.30}
route_b = chrono.fit(phase_observations=[obs], tz=TZ)
print("\n--- Route B ---\n" + route_b.summary())

# 3. Route C with dynamic phase
route_c = chrono.fit(sleep=sleep, phase_observations=[obs], tz=TZ, dynamic=True)
print("\n--- Route C (smoother) ---\n" + route_c.summary())
tr = route_c.trajectory.merge(truth[["date", "dlmo_h"]].rename(columns={"dlmo_h": "true_h"}), on="date")
print(tr[["date", "dlmo_clock", "sd_h", "true_h"]].assign(true=lambda d: d.true_h.map(chrono._time.fmt_clock))
      .drop(columns="true_h").to_string(index=False))

# 4. Align a CGM-like series
t = pd.date_range("2026-03-09 00:00", "2026-03-11 00:00", freq="15min", tz=TZ)
cgm = pd.DataFrame({"datetime": t, "glucose": 95 + 8 * np.sin(2 * np.pi * (t.hour + t.minute / 60) / 24)})
aligned = chrono.transform(cgm, route_c, reference=["phase", "chronotype", "wake"], circular_encoding=True)
print("\n", aligned.iloc[36:42, :9].round(2).to_string(index=False))
print("\npsi diagnostics:", chrono.psi_diagnostics(aligned))

# 5. Schedules
print("\nWake-anchored (predicted):\n", chrono.schedule(route_c, anchor="wake", offsets=[2, 7, 12],
      dates=["2026-03-10", "2026-03-14"], mode="predicted", last_before_sleep=1)[["date", "day_type", "clock", "status"]].to_string(index=False))
print("\nPhase-anchored:\n", chrono.schedule(route_c, anchor="phase", offsets=[10, 16], dates=["2026-03-10"])
      [["anchor_date", "offset_h", "clock", "lower", "upper", "anchor_source"]].to_string(index=False))
