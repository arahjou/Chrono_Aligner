# ChronoAlign

ChronoAlign is a Python package for expressing an observation's time relative to a **person's own sleep and circadian timing**. It takes sleep episodes, light exposure, and/or an existing phase measurement; builds a `CircadianProfile`; then adds relative-time columns to data such as glucose, temperature, activity, or cognitive measurements. It also calculates clock times for events scheduled relative to wake or circadian phase.

For example, suppose someone's sleep-based chronotype marker is 04:00, estimated melatonin onset is 21:00 the previous evening, and wake is 07:00. Their 08:00 glucose reading is then 4 hours after the chronotype marker, 11 hours after estimated melatonin onset, and 1 hour after waking. Those coordinates answer different questions. ChronoAlign keeps them separate, identifies the source of a physiological estimate, and carries its uncertainty into transformed data and schedules.

## The idea

Clock time alone does not tell us where a person is in their sleep/wake cycle or internal circadian cycle. ChronoAlign uses three distinct references:

| Reference | What it means | What it needs |
|---|---|---|
| **Behavioural chronotype (CRT)** | Hours relative to MSFsc, the sleep corrected midpoint of sleep on free days. This describes observed sleep timing, not melatonin phase. | Sleep onset/end plus work/free-day and alarm information |
| **Physiological phase** | Hours relative to measured dim-light melatonin onset (**DLMO**) or a predicted DLMO (**pDLMO**). The source is named, such as `pDLMO[HairTime]` or `pDLMO[light_model]`. | A phase observation or a light time series |
| **Wake-dependent time** | Hours since the last recorded wake and the duration of the preceding sleep episode. | Sleep episodes |

The package adds these coordinates to each input row; it does **not** move or replace the original timestamp. Phase and chronotype coordinates have a 0–24-hour wrapped form (`*_phase_h`) and an unbounded form relative to a dated reference instance (`*_linear_h`). Optional sine/cosine columns help model a 24-hour cycle without a discontinuity at the wrap point.

The distinction matters in analysis: people can share the same sleep timing yet have different physiological phases. Time since wake and phase-relative time can also track closely within a day; `psi_diagnostics()` and related functions help check whether a dataset can separate their effects.

## What it does in practice

1. **Build a profile with `fit()`.** Sleep records yield sleep duration, midsleep, social jetlag, regularity, and, when work/free-day labels are available, MSFsc. A supplied DLMO or assay result supplies a physiological anchor. The built-in light model can predict daily phase from `datetime`/`lux` data. With phase evidence and `dynamic=True`, a state-space model estimates phase across days, using sleep and/or light data when available.
2. **Align observations with `transform()`.** A copy of your data gains requested columns such as `crt_phase_h`, `pdlmo_phase_h` or `dlmo_phase_h`, `hours_since_wake`, `prior_sleep_duration_h`, `phase_uncertainty_h`, and `anchor_source`. You can also add time since supplied events such as meals.
3. **Plan times with `schedule()` and `schedule_phase_sample()`.** Compute events at offsets from a predicted or logged wake, at offsets from a dated DLMO/pDLMO, or find a suitable time to collect a phase assay sample. Phase-based schedule output includes the anchor source and propagated time uncertainty.

Available coordinates depend on the inputs:

| Route | Inputs | Available result |
|---|---|---|
| **A** | Sleep only | Wake-dependent coordinates; also behavioural chronotype when work/free-day and alarm labels support MSFsc. No physiological phase |
| **B** | Phase observation and/or light, without sleep | Physiological phase coordinates; no wake-dependent or MSFsc coordinate |
| **C** | Sleep plus a phase observation and/or light | Physiological and wake-dependent coordinates; also behavioural chronotype when MSFsc is available. Phase angles and optional daily phase estimation |

`fit()` reports a route and an evidence grade. A grade describes the **kind of evidence supplied**, not guaranteed accuracy. Requesting a coordinate unsupported by the profile raises an explanatory error.

## Install and try it

From this repository:

```bash
pip install -e .
python examples/demo.py
```

This self-contained example uses an **existing HairTime result** to align two readings. `PhaseObservation` accepts the assay's predicted DLMO clock time and the exact sample time; ChronoAlign does not run the hair assay itself.

```python
import pandas as pd
import chronoalign as chrono

observation = chrono.PhaseObservation(
    source="HairTime",
    estimate_clock="21:10",  # assay-predicted DLMO
    sampled_at="2026-03-04T10:15:00+01:00",
    day_type="workday",
)
profile = chrono.fit(
    phase_observations=[observation], tz="Europe/Berlin"
)

readings = pd.DataFrame({
    "datetime": pd.to_datetime([
        "2026-03-04T08:00:00+01:00",
        "2026-03-04T12:00:00+01:00",
    ]),
    "glucose": [95, 101],
})
aligned = chrono.transform(
    readings, profile, reference="phase", circular_encoding=True
)
print(profile.summary())
print(aligned[[
    "datetime", "glucose", "pdlmo_phase_h",
    "phase_uncertainty_h", "anchor_source",
]])

# The sample refers to the DLMO of the previous evening (3 March).
plan = chrono.schedule(
    profile, anchor="phase", dates=["2026-03-03"], offsets=[10, 16]
)
print(plan[["anchor_date", "clock", "sd_h", "anchor_source"]])
```

This is **Route B**. Applying its single phase estimate to another day assumes stable phase; `profile.summary()` states that assumption. To add behavioural and wake-dependent coordinates, pass a `sleep` DataFrame to `fit()` and request `reference=["phase", "chronotype", "wake"]`. The runnable [demo](examples/demo.py) shows Routes A, B, and C together on synthetic data.

## Input and output details

| Input | Expected data |
|---|---|
| `sleep` | One row per main sleep episode: timezone-aware `sleep_onset` and `sleep_end`. Add `alarm` and `work_day` to calculate MSFsc; `work_day` describes the **day after** that sleep. Optional `shift_type` identifies shift schedules. |
| `light` | Time series with `datetime` and `lux`. About nine or more days are recommended for the light model. The current model uses light, not activity. |
| `phase_observations` | `PhaseObservation` objects with `source` (`DLMO`, `HairTime`, `BodyTime`, or `light_model`), an exact `sampled_at`, and either `estimate_clock` or `internal_time_h`. Optional QC fields and `uncertainty_h` affect the reported uncertainty. |
| Data for `transform()` | A pandas DataFrame with a timestamp column (default `datetime`) and any measurement columns to retain. |

A morning assay sample is linked to the **previous evening's** DLMO when appropriate. A measured DLMO is labelled `DLMO`; assay and light-model estimates remain labelled `pDLMO[source]`. The profile stores dates, sources, uncertainty, flags, and disagreements between observations. For a fixed phase anchor, extending it to other days is an explicit stability assumption. For dynamic phase, the default smoother uses observations from the whole fitted period; `smoother=False` selects a causal filter that uses only data available up to each day.

## Scope and limitations

- This package consumes HairTime and BodyTime **results**; it does not process biological samples or produce their assay predictions.
- Sleep timing is a behavioural reference. It is not converted into measured DLMO. A sleep-based prior used to plan assay collection relies on an explicit population phase-angle assumption.
- Assay validity depends on population, sample timing, and QC. Default QC thresholds and error values need calibration for inferential use. The light model and the synthetic validation in this repository do not establish real-world individual accuracy.
- The scheduler calculates requested relative clock times. It does not decide clinically appropriate medication doses or dosing schedules.

See the [implementation report](ChronoAlign_implementation_report.md) for design decisions and synthetic validation, and the [v2 specification](ChronoAlign_v2_multisource_anchors.docx) for the underlying concept. To run the tests, install `pip install -e ".[test]"` and run `pytest`.
