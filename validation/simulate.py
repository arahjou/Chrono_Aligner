"""Synthetic validation of phase-estimation strategies in ChronoAlign.

Question: given 21 days of sleep records and one or two single-sample phase
observations, how well does each strategy recover DLMO on *every* evening, and are
its uncertainty intervals calibrated?

Run:  python validation/simulate.py [n_participants]
"""
import sys
import warnings

import numpy as np
import pandas as pd

import chronoalign as chrono
from chronoalign.synthetic import SimConfig, simulate_observation, simulate_participant

warnings.filterwarnings("ignore")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
TZ = "Europe/Berlin"
SAMPLE_WED = 8      # evening index 8 = Tuesday week 2 -> HairTime sample Wednesday 10:00
SAMPLE_SAT = 12     # Saturday evening -> sample Sunday 10:00


def evaluate(truth, prof, strategy, scenario, pid):
    rows = []
    for i, (_, r) in enumerate(truth.iterrows()):
        h, sd = prof.dlmo_for_date(r["date"])
        err = float(chrono._time.wrap12(h - r["dlmo_h"]))
        rows.append(dict(scenario=scenario, pid=pid, strategy=strategy, date=r["date"], free_evening=r["free_evening"],
                         shifted=r["shifted"], err=err, sd=sd, covered=abs(err) <= 1.96 * sd, day_index=i))
    return rows


def run(scenario, cfg):
    out = []
    for pid in range(N):
        sim = simulate_participant(pid, cfg)
        truth, sleep = sim["truth"], sim["sleep"]
        rng = np.random.default_rng(10_000 + pid)
        h1 = simulate_observation(truth, SAMPLE_WED, rng=rng)
        h2 = simulate_observation(truth, SAMPLE_SAT, rng=rng)
        dl = simulate_observation(truth, SAMPLE_WED, source="DLMO", error_sd_h=0.5, rng=rng)
        strategies = {
            "HairTime x1, fixed (stability assumption)": dict(phase_observations=[h1]),
            "HairTime x1 + sleep, filter (causal)": dict(phase_observations=[h1], dynamic=True, smoother=False),
            "HairTime x1 + sleep, smoother": dict(phase_observations=[h1], dynamic=True),
            "HairTime x2, fixed pooled": dict(phase_observations=[h1, h2]),
            "HairTime x2 + sleep, smoother": dict(phase_observations=[h1, h2], dynamic=True),
            "Measured DLMO x1, fixed": dict(phase_observations=[dl]),
            "Measured DLMO x1 + sleep, smoother": dict(phase_observations=[dl], dynamic=True),
        }
        for i, (_, r) in enumerate(truth.iterrows()):   # no individual data at all
            err = float(chrono._time.wrap12(cfg.base_mean_h - r["dlmo_h"]))
            out.append(dict(scenario=scenario, pid=pid, strategy="Population mean only (no data; simulator SD 1 h)",
                            date=r["date"], free_evening=r["free_evening"], shifted=r["shifted"], err=err,
                            sd=cfg.base_sd_h, covered=abs(err) <= 1.96 * cfg.base_sd_h, day_index=i))
        for name, kw in strategies.items():
            obs = [chrono.PhaseObservation(**{k: getattr(o, k) for k in
                   ("source", "estimate_clock", "sampled_at", "internal_time_h", "day_type")}) for o in kw.pop("phase_observations")]
            prof = chrono.fit(sleep=sleep, phase_observations=obs, tz=TZ, **kw)
            out += evaluate(truth, prof, name, scenario, pid)
    return pd.DataFrame(out)


def main():
    res = pd.concat([
        run("entrained (workday/free-day shifts only)", SimConfig()),
        run("30% with a sustained 2-h delay from day 12", SimConfig(shift_prob=0.3)),
    ], ignore_index=True)
    res["abs_err"] = res["err"].abs()
    summ = (res
            .groupby(["scenario", "strategy"], sort=False)
            .agg(MAE_h=("abs_err", "mean"), P90_abs_err_h=("abs_err", lambda x: np.quantile(x, 0.9)),
                 bias_h=("err", "mean"), mean_sd_h=("sd", "mean"), coverage95=("covered", "mean"))
            .round(2))
    shifted = (res[res["shifted"]]
               .groupby(["scenario", "strategy"], sort=False)["abs_err"].mean().round(2).rename("MAE_shifted_participants_h"))
    later = (res[res["day_index"] >= SAMPLE_WED]
             .groupby(["scenario", "strategy"], sort=False)["abs_err"].mean().round(2).rename("MAE_from_sample_day_h"))
    summ = summ.join(later).join(shifted)
    pd.set_option("display.width", 200)
    print(summ.to_string())
    summ.to_csv("validation/simulation_summary.csv")
    res.to_csv("validation/simulation_rows.csv.gz", index=False)


if __name__ == "__main__":
    main()
