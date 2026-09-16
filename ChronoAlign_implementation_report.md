# ChronoAlign v2: implementation report

**Source specifications:** *ChronoAlign_v1* and *ChronoAlign_v2_multisource_anchors* (Google Drive, ChronoAligner folder)
**Deliverable:** Python package `chronoalign` 0.2.0, with 36 passing tests, a demo, and a synthetic validation study
**Date:** 16 September 2026

---

## 1. Mission

The goal was to turn the v2 concept into working software. The engine converts clock time into personalised coordinates: behavioural (CRT), physiological (DLMO or pDLMO) and wake-dependent. It accepts any combination of sleep records, light data and single-sample phase assays. Every output records its source, the date it refers to and its uncertainty. The engine also schedules events and sampling from these coordinates, and says what each input route cannot provide.

All v1 corrections are kept: the MCTQ MSFsc rule, circular statistics with UTC arithmetic, split phase and linear CRT, separate precision and validity, retrospective-only `time_until_sleep`, identifiability diagnostics, and two scheduling modes.

## 2. What was built

| v2 section | Implementation | Status |
|---|---|---|
| §1 Three input routes | `fit()` detects Route A, B or C. Asking for a coordinate that a route cannot give raises an error that lists what the route can and cannot provide. | Done |
| §2 Anchor sources | `SOURCES` table: DLMO, HairTime, BodyTime, light_model, each with a default SD, a validated window and a validity domain | Done |
| §3 Sampling window | `schedule_phase_sample()`. It takes a population, sleep-based or phase prior, and returns the robust window, a recommended time and the probability of landing in the window. | Done |
| §4 PhaseObservation | Converts the timestamp to a DLMO instant (sampling time minus internal time) and labels it with its evening date. It checks the validated window, widens the SD from QC indicators, and adds flags. | Done |
| §4 Sleep phenotype | MSW, MSF, MSFsc, SJL, SJLsc, weekly sleep loss, circular SDs, SRI, MCTQShift-style free-day midsleep | Done |
| §4 Combining rules | Behavioural and physiological scales are kept separate. Source tiers are DLMO first, then assay, then light model. Same-tier observations are pooled by inverse-variance circular weighting. Disagreements are reported, not averaged away. | Done |
| §5 Transform | `crt_phase_h`, `crt_linear_h`, `(p)dlmo_phase_h`, `(p)dlmo_linear_h`, `phase_uncertainty_h`, `anchor_source`, `phase_mode`, `hours_since_wake`, `prior_sleep_duration_h`, sin/cos encodings, time since an event (e.g. meals). Original timestamps are left untouched. | Done |
| §6 Dynamic phase | Kalman filter and RTS smoother over the state [DLMO_d, ψ_free, ψ_work], with light-model drift, per-source measurement noise, noisier alarm nights, and wrapped innovations | Done |
| §7 Identifiability | `psi_table`, `psi_diagnostics` (SD of ψ), a concurvity proxy, and `sensitivity_analysis` (anchor perturbed within its SD) | Done. The GAMM itself is left to mgcv. |
| §8 Scheduling | Wake-anchored (predicted or wake-triggered, with drop, flag or compress rules), constrained spacing, phase-anchored with propagated uncertainty, and assay sampling | Done |
| §9–10 Architecture and API | Python API as sketched in v2 (`chrono.fit / transform / schedule / schedule_phase_sample`) | Python done; R not started |
| §11 Evidence grades | S0, S1, M, P1, P2, C, assigned from the inputs actually used | Done |
| Light model | Forger–Jewett–Kronauer 1999 model with DLMO = CBTmin − 7 h. A 1-h change in the light schedule gives a 1.0-h change in DLMO. A test compares it with the `circadian` package to within 0.15 h. | Done (light only; activity not used) |

**Every worked example in the v2 document is reproduced in the test suite:**

* 13.1 h after pDLMO
* the 08:00–12:00 sampling window
* the CRT +4 / pDLMO +11.5 / +9.5 table
* 09:15, 14:15 and 19:15 for a 07:15 wake
* the 00:00 event flagged for a 10:00 waker
* the 09:00, 14:00, 19:00 constrained schedule
* the within-episode identity DLMO_linear = hours_since_wake + ψ

The suite also tests a sleep episode that crosses a DST change (23:00 CET to 07:00 CEST counts as 7 h).

## 3. Issues in the v2 text, and the decisions taken

1. **Which calendar day does a pDLMO belong to?** In the v2 profile example, `phase_anchor_date = "2026-03-04 (workday)"`. But a 10:15 sample taken 13.1 h after pDLMO refers to the DLMO of the evening of **3 March**. The engine labels each DLMO by its evening date and stores the sample's day type separately.
   *This matters for weekends:* a Sunday-morning hair sample describes Saturday evening, which is before a free day. If workday and free-day phase both matter (§3), the plan should be based on the evening, not the sampling day.
2. **ψ_d = wake_d − DLMO_d is ambiguous.** It could mean the DLMO of the same calendar day (after the wake) or the preceding evening. The engine uses the preceding DLMO instance, so ψ is about +9 to +10 h. The identity with `hours_since_wake` then holds within each wake episode.
3. **The example's `uncertainty_h = 1.0` understates HairTime error.** The validation gives a median absolute error of 0.99 h and limits of agreement of −2.84 to +2.90 h. Both imply an SD of about **1.46 h** under normality. The engine uses 1.46 h by default. The value 1.0 is closer to the median absolute error than to an SD.
4. **Threshold for flagging disagreement.** "More than their combined uncertainty" means 1 SD. That would flag about 32% of pairs that actually agree. The default is z = 1.96, and it can be changed.
5. **Phase-angle prior in the state-space model.** A tight prior on ψ (e.g. midsleep ≈ DLMO + 6 ± 1.5 h) quietly turns sleep timing into a DLMO level. In testing, it moved a HairTime anchor by 34 min and shrank its SD from 1.5 to 0.9 h. That contradicts the §4 rule not to average MSFsc with pDLMO. The prior is therefore diffuse by default, so the phase level comes only from phase observations and sleep informs changes.
6. **The causal filter must not look ahead.** Starting the filter from a later observation leaks future information. The filter now starts from a population prior unless an observation exists on the first day.
7. **Alarm on free days.** v2 keeps the questionnaire rule (MSFsc = NA). For diary data the engine follows v1: free days with an alarm are excluded from MSF, and MSFsc is NA only if fewer than `min_free_days` alarm-free free days remain.
8. **QC thresholds are not specified.** The `amplitude_score` scale and cut-off are unknown, so no inflation is applied until `QCPolicy.amplitude_threshold` is set, and a flag says so. Between-model disagreement (`model_sd_h`) is added in quadrature. These defaults are placeholders.
9. **Light model.** Woelders et al. (2017) used light *and* activity. The engine uses light only. Daily model DLMOs are strongly autocorrelated, so when fused their SD is inflated by √n_days, and day-to-day changes enter as drift.
10. **Minor:** the v1 constrained example is correct for the times it states. The spacing constraint alone could be met with sleep onset from 19:00, and the engine reports both.

## 4. Synthetic validation

**Setup:**

* 200 simulated participants per scenario, 21 evenings each
* True DLMO has a trait component (between-person SD 1 h), a later shift on free evenings that is larger in late types, and gradual adaptation
* Sleep follows DLMO, and workday alarms truncate it
* HairTime error SD is 1.46 h and measured-DLMO error SD is 0.5 h
* Sample 1 is on a Wednesday morning; sample 2 is on a Sunday morning

The error is computed on **every evening**, not only the sampling day.

| Strategy | MAE, entrained (h) | 95% coverage, entrained | MAE, 30% with a 2-h delay (h) | 95% coverage, delay scenario |
|---|---|---|---|---|
| Population mean only, no data | 0.87 | 0.94 | 1.00 | 0.89 |
| HairTime ×1, fixed (stability assumption) | 1.16 | 0.97 | 1.28 | 0.94 |
| HairTime ×1 + sleep, smoother | 1.13 | 0.98 | 1.18 | 0.96 |
| HairTime ×2, fixed pooled | 0.90 | 0.95 | 0.99 | 0.91 |
| **HairTime ×2 + sleep, smoother** | **0.88** | 0.96 | **0.91** | 0.95 |
| Measured DLMO ×1, fixed | 0.45 | 0.93 | 0.66 | **0.80** |
| Measured DLMO ×1 + sleep, smoother | 0.45 | 0.98 | 0.56 | 0.92 |

**Sensitivity to the spread of DLMO between people** (120 participants, entrained):

| Between-person DLMO SD | Population mean | HairTime ×1 | HairTime ×2 + sleep |
|---|---|---|---|
| 0.75 h | 0.67 | 1.24 | 0.86 |
| 1.0 h | 0.88 | 1.24 | 0.85 |
| 1.5 h | 1.31 | 1.24 | 0.84 |
| 2.0 h | 1.73 | 1.24 | 0.84 |

**What this shows:**

* **One assay sample only beats the population mean when people differ by more than the assay error.** With an SD of about 1.46 h, a single HairTime sample adds individual information mainly in heterogeneous samples or for extreme chronotypes. This supports the v2 point that a 2-h individual difference is "within assay error; at group level informative."
  *Suggested addition:* an optional population-prior (shrinkage) estimate, clearly labelled.
* **Two samples, fused with sleep, are the most robust design.** The MAE was about 0.85–0.91 h regardless of population spread or phase shifts, with calibrated intervals. This supports the Route C recommendation to repeat samples.
* **The stability assumption costs calibration even with gold-standard DLMO.** Interval coverage fell to 0.80 when real phase shifts occurred. The smoother restored it to 0.92 and cut MAE by 15%.
* **Pooling a workday and a free-day sample** adds a small bias toward the free-day phase (+0.15 h). Use the smoother or day-type-specific anchors instead.

**Caveat:** the simulator and the state-space model share structure (sleep follows DLMO through a phase angle). These results show that the software behaves correctly and that its uncertainty is internally consistent. They are not evidence of real-world accuracy.

## 5. Recommended next steps

1. **Calibrate `QCPolicy` and the HairTime error model on real validation data** (model SD, amplitude, sampling time against absolute error), if you can access the HairTime validation cohort. This is the biggest open parameter.
2. **Validate against measured DLMO in real data**, first in regular day workers, then in shift workers. Compare Routes A, B and C.
3. **Add an optional population-prior estimate**, labelled for example `pDLMO[HairTime|pop]`, and report both values.
4. **Build the R interface.** Options are a thin reticulate wrapper, or a native port that reuses `mctq` and delegates §7 models to `mgcv`.
5. **Add activity input to the light model**, following Woelders et al., and fuller MCTQShift handling of rotating rosters.
6. **Choose the repository, licence and name.** The note uses "ChronoAlign", but the project is called ChronoAligner.
