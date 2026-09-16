"""Does one assay sample beat the population mean? Depends on between-person DLMO spread."""
import warnings

import numpy as np

import chronoalign as chrono
from chronoalign._time import wrap12
from chronoalign.synthetic import SimConfig, simulate_observation, simulate_participant

warnings.filterwarnings("ignore")
for bsd in (0.75, 1.0, 1.5, 2.0):
    cfg = SimConfig(base_sd_h=bsd)
    e = {"pop": [], "h1": [], "h2s": []}
    for pid in range(120):
        sim = simulate_participant(pid, cfg)
        tr = sim["truth"]
        rng = np.random.default_rng(5000 + pid)
        h1, h2 = simulate_observation(tr, 8, rng=rng), simulate_observation(tr, 12, rng=rng)
        p1 = chrono.fit(sleep=sim["sleep"], phase_observations=[h1])
        h1b = chrono.PhaseObservation(source="HairTime", internal_time_h=h1.internal_time_h, sampled_at=h1.sampled_at)
        p2 = chrono.fit(sleep=sim["sleep"], phase_observations=[h1b, h2], dynamic=True)
        for _, r in tr.iterrows():
            e["pop"].append(abs(wrap12(cfg.base_mean_h - r.dlmo_h)))
            e["h1"].append(abs(wrap12(p1.dlmo_for_date(r.date)[0] - r.dlmo_h)))
            e["h2s"].append(abs(wrap12(p2.dlmo_for_date(r.date)[0] - r.dlmo_h)))
    print(f"between-person DLMO SD {bsd:>4} h | MAE population mean {np.mean(e['pop']):.2f} | "
          f"HairTime x1 fixed {np.mean(e['h1']):.2f} | HairTime x2 + sleep smoother {np.mean(e['h2s']):.2f}")
