# ChronoAlign 0.2 — a circadian reference engine (Python)

Implements the *ChronoAlign v2* concept (multi-source phase anchors). Every timestamp gets
personalised coordinates, and each coordinate carries its **source, the date it refers to, and its uncertainty**.

```bash
pip install -e ".[test]"
pytest                          # 36 tests, including every worked example in the v2 document
python examples/demo.py         # Routes A, B, C on a synthetic participant
python validation/simulate.py   # synthetic comparison of phase-estimation strategies (~1 min)
```

```python
import chronoalign as chrono

obs = chrono.PhaseObservation(source="HairTime", estimate_clock="21:10",
                              sampled_at="2026-03-04T10:15:00+01:00", day_type="workday",
                              qc={"model_sd_h": 0.35, "amplitude_score": 3.1})
profile = chrono.fit(sleep=sleep, light=None, phase_observations=[obs],
                     tz="Europe/Berlin", dynamic=True)        # state-space smoother
print(profile.summary())

aligned = chrono.transform(cgm, profile, timestamp="datetime",
                           reference=["phase", "chronotype", "wake"], circular_encoding=True)
chrono.psi_diagnostics(aligned)                               # can f(phase) and g(time awake) be separated?

chrono.schedule(profile, anchor="wake", offsets=[2, 7, 12], mode="wake_triggered", wake_time="2026-03-10T07:15+01:00")
chrono.schedule(profile, anchor="phase", offsets=[10, 16], dates=["2026-03-10"])
chrono.schedule_phase_sample(assay="HairTime", prior=profile)
```

## Inputs

| Input | Columns / fields |
|---|---|
| `sleep` | `sleep_onset`, `sleep_end` (tz-aware), `alarm`, `work_day` (day *after* the sleep is a workday), optional `shift_type` |
| `light` | `datetime`, `lux` (≥ ~9 days recommended) |
| `PhaseObservation` | `source` ∈ DLMO, HairTime, BodyTime, light_model; `estimate_clock` **or** `internal_time_h`; exact `sampled_at` with UTC offset; `day_type`; `qc`; optional `uncertainty_h` |

## Modules ↔ v2 sections

| Module | Spec | Content |
|---|---|---|
| `sleep.py` | §4 | MSW/MSF/MSFsc (MCTQ rule), SJL, SJLsc, weekly sleep loss, circular SDs, SRI, MCTQShift-style free-day midsleep, schedule type |
| `phase.py` | §2, §4 | `PhaseObservation` (timestamp → DLMO instant, evening date, validated window, QC-widened SD), inverse-variance circular pooling, disagreement report |
| `profile.py`, `grades.py` | §1, §4, §11 | `CircadianProfile`, routes A/B/C with explicit capabilities, evidence grades S0/S1/M/P1/P2/C |
| `statespace.py` | §6 | Kalman filter / RTS smoother over [DLMO_d, ψ_free, ψ_work] with light drift, assay/DLMO/sleep measurement models |
| `light.py` | §2, §6 | Forger–Jewett–Kronauer (1999) model, DLMO = CBTmin − 7 h; cross-checked against the `circadian` package |
| `transform.py` | §5, §12 | `crt_phase_h/crt_linear_h`, `dlmo_/pdlmo_phase_h` and `_linear_h`, `hours_since_wake`, `prior_sleep_duration_h`, `daily_sleep_phase_h`, event-relative time, sin/cos |
| `identifiability.py` | §7 | ψ table, ψ SD, concurvity proxy, anchor-perturbation sensitivity analysis |
| `schedule.py` | §8 | predicted / wake-triggered, constrained spacing, phase-anchored with propagated uncertainty, assay-sampling window |
| `synthetic.py` | — | simulator with known true DLMO (software validation only) |

## Conventions

* Durations use UTC arithmetic; clock summaries are circular. Naive timestamps on a DST transition raise.
* A DLMO instance is labelled by its **evening date** (instant − 12 h). A morning hair sample refers to the previous evening's DLMO.
* ψ (phase angle) = sleep onset / midsleep / wake of the episode **following** a DLMO instance, minus that instance.
* Predicted markers are always `pDLMO[source]`; a predicted value can never be labelled `DLMO`.
* Default QC thresholds are placeholders (see `QCPolicy`) and must be calibrated before inferential use.
* The engine computes requested phase-relative times; it does not determine clinical dosing schedules.
