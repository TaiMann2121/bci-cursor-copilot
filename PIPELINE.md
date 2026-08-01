# The Copilot Pipeline: architecture decision and the credibility argument

This document states the final pipeline for the arm-BCI copilot and the argument
for why its improvements are likely to carry over to real human trials. Read it
after `README.md`; for a code-reading order see `READING_GUIDE.md`. Code details
live in the module docstrings; the empirical history lives in `Progress Reports/`.

> **Status of this document (updated 7/29/2026).** The architecture argument in
> §3–§5 stands. The *measurements* it was originally built on do not: a loader
> defect (§2.5) corrupted the real-data trial set behind every pre-7/29 number, and
> the corrected raw-decoder baseline is **64.0%**, not the ~53–55% quoted below and
> in earlier reports. §2 now marks each claim by whether it survives. §7 sorts the
> standing conclusions; §8 gives the current next steps. Do not quote a number from
> this file that §7 lists as *pending re-run*.

---

## 1. The problem (from the project brief)

In the two-stage typing paradigm, the **arm phase** drives a cursor from center
toward one of 8 directions; EEGNet predicts per-tick cursor velocities, and the
**final cursor position** selects the direction. Single-trial EEG is noisy, so the
brief asks for a **copilot**: a model that *observes the decoded trajectory,
infers the intended target, and aids the cursor toward it* — the shared-autonomy
idea established in Lee et al., *Nature Machine Intelligence* 2025 (the paper the
brief links), where AI copilots raised BCI cursor hit-rate by 2.1–3.9×.

Our copilot is a supervised **LSTM target classifier** (per-tick, 8-way) whose
belief drives an **additive corrective velocity** toward the predicted target
(`copilot_core.py`). That control law is settled. The open question has always
been: **how do we train and evaluate it so a gain is real and transfers to
humans?**

---

## 2. What a month of experiments actually settled

The work converged through a disciplined funnel (see `Progress Reports/`), and it
is worth being clear about what is *known*, because it directly determines the
final architecture:

1. **The control law works, but the open-loop gain is small.** ✅ *Survives, and is
   re-measured.* A within-subject copilot improves endpoint accuracy by roughly
   **+1 pp** over the raw decoder. On the fixed loader and held-out test blocks
   (3 seeds): raw inputs **+1.10 ± 0.30 pp**, cleaned inputs **+1.43 ± 0.16 pp**,
   over a 62.7% raw-decoder baseline on those blocks (64.0% over all trials).

2. **Training-data composition has been mined out.** ⚠️ *Pending re-run.* The
   finding was: real-only is a strong baseline; calibrated EEGK-sim helps only as a
   **~25% minority augmenter** (+0.9 pp, leakage-free, 3 seeds); higher sim
   fractions and pure sim are worse; the **endpoint-modeled surrogate does not help
   as training data** because it carries target-*independent* structure the copilot
   cannot exploit. All of it was measured on the corrupted trial set (§2.5).

3. **Open-loop correctability is capped.** ⚠️ *Pending re-run, and partly
   contradicted.* The residual diagnostic reported ~87% of the decoder's wrong
   trials carry *no recoverable intent* in the replayed trajectory, with the model's
   belief pointing at the true target on ~28% of wrong trials but only ~8%
   converting into the endpoint. Two caveats now attach: it was measured on the
   corrupted trial set, and it never measured whether the classifier *learns* — the
   7/29 learning diagnostic (`studies/diagnose_learning.py`) shows it plainly does
   (train CE far below chance, validation tracking, 57–78% final-tick target
   accuracy vs 12.5% chance). What remains solid is that smarter **open-loop**
   steering (`trajectory_aware_copilot.py`) **fails**: causal accumulated-belief
   steering never beats the instantaneous copilot, and steering harder makes it
   monotonically worse (−0.18 to −2.26 pp).

**These facts point at one conclusion, and it is the crux of this document.** The
re-runs in §8 can change the magnitudes; the structural argument in §3 does not
depend on them.

---

## 2.5. The data-integrity fix that invalidated the numbers (7/29)

Trials are keyed by `(subject, session_number, run_number, trial_number,
inner_trial_number)`. **Those keys are not unique across session folders** — S01's
WordTyping and SentenceTyping sessions both contain `(0, 1, 1, 1)`, with different
target labels. The production loader concatenated all 22 session CSVs and grouped by
those keys alone, silently fusing **3,030 pairs of distinct trials into mixed-target
sequences** and collapsing 16,197 real trials into 13,167 corrupted ones.

**The fix (now in `copilot_dataset.py`):** trajectories carry their source session
(`session_id`); real data is loaded per session folder and never concatenated
(`_load_eegk_real_sessions`); `split_real` blocks by session folder. Verified: raw
and cleaned loads give identical trial counts, identical held-out test sets, and
identical baselines, with zero train/test leakage. This fix is **not optional** — the
previous behavior produced corrupted trials, so pre-7/29 runs will not reproduce
their original numbers, by design.

**Alongside it, metric-safe input cleaning** (`clean_trajectory_sessions`): trim the
leading pre-onset dead ticks, then rescale each session so its median final radius
maps onto a common per-subject reference. Both steps preserve direction, so the
endpoint metric — and therefore every label — is unchanged; only what the copilot
*reads* changes. Trimming carries the benefit (+0.17 pp alone, and it halves the
variance); rescaling alone is slightly negative (−0.24 pp) and should never be
applied without trimming; together they give +0.33 pp. Cleaning is currently
**opt-in** (`load_source(clean=True)`) and is not yet wired into `train_copilot.py`
or `evaluate_copilot.py`.

---

## 3. The diagnosis: the open-loop *evaluation* is the bottleneck

Every result above was measured **open-loop**: a recorded BCI velocity stream is
replayed and the copilot's correction is added on top. The recorded velocities
never react to where the copilot moves the cursor.

But the copilot's entire value — in Lee et al. and in any real deployment — is a
**feedback** effect. When the copilot nudges the cursor toward the inferred goal,
the *user keeps steering toward the target from the new position*, so an early
correct nudge **compounds**: straighter paths, faster dial-in, more hits. Open-loop
replay cannot represent this, because the "user" (a fixed recording) never
responds. This is exactly why smarter open-loop steering fails: with the recording
frozen, steering harder just adds variance and overshoots.

So the ~87% "uncorrectable" ceiling and the belief→endpoint gap are (pending the
§8 re-run of that diagnostic) **artifacts of
open-loop replay**, not fundamental limits. The paradigm has both *plateaued* and
started to *actively mislead*.

**This also reconciles the apparent contradiction in the recent reports.** The
7/13 recommendation "do not build the surrogate" was about the surrogate **as
training data** — and that recommendation stands. It says nothing about a
surrogate **as a closed-loop environment**, which is a different artifact serving
a different purpose. Conflating the two is what made the path forward feel stuck.

---

## 4. The decision: a closed-loop pipeline

The final pipeline evaluates (and can train) the copilot **in closed loop against
a subject-level surrogate**, mirroring the methodology of the paper the brief
cites.

```
          ┌─────────────────────── closed loop ───────────────────────┐
          │                                                            │
   target ─▶  SUBJECT SURROGATE ──bci_vel_t──▶  COPILOT ──correction──▶ cursor_{t+1}
          │   (per-subject decoder-           (LSTM belief →           │
          │    emulator, re-aims from          additive corrective     │
          └─── current cursor each tick)       velocity)  ─────────────┘
                                                            ▲
                                              cursor_{t+1} feeds back
```

**Surrogate** (`closed_loop.py`). A per-subject generative model of the decoder's
per-tick output as **noisy, target-directed control from the *current* cursor**:
`heading_t = angle(target − cursor_t) + N(0, σ)`, magnitude drawn from the
subject's real step-magnitude texture, with a real-sampled onset dwell. The single
fitted parameter σ is calibrated so that, run open-loop, the surrogate reproduces
that subject's **real endpoint accuracy**. Because it re-aims from the current
cursor, the copilot's displacement changes the next command — the feedback channel
is present. This is structurally the Lee et al. training environment, calibrated to
our EEGK subjects.

**Copilot.** The existing LSTM + additive corrective velocity, now evaluated in the
loop. Two upgrades the closed-loop setting unlocks (both already seeded in the
repo):
  - **Trajectory-aware steering** (`trajectory_aware_copilot.py`): causal
    accumulated belief. It failed open-loop; the hypothesis is that its early,
    decisive steering only pays off when the loop is closed.
  - **Key-prior conditioning** (Problem 2 → Problem 1 bridge): condition the
    target inference on a **language-model prior over the next direction** given
    previously typed keys, so the copilot's belief is sharper before the cursor
    even moves. Seed: `experimental/fusion_pipeline.py`.

---

## 5. Why closed-loop-on-surrogate is credible evidence for humans

A surrogate result is only as trustworthy as the surrogate. The pitch is not "it
improved in my simulator"; it is a **three-legged validation** that makes transfer
falsifiable, plus a published precedent.

- **Leg 1 — Faithfulness (open-loop).** The surrogate reproduces each subject's
  real endpoint accuracy *by calibration*; its **angle-error distribution and
  per-direction accuracy are not fitted**, so their agreement with real data is
  independent evidence the emulator is faithful.
- **Leg 2 — Copilot consistency.** Run the *same* trained copilot open-loop on
  real held-out data and on the surrogate. If its Δaccuracy agrees, the surrogate
  is not handing the copilot artificial structure — the precondition for trusting
  its closed-loop behavior.
- **Leg 3 — Closed-loop payoff.** Only after legs 1–2 hold is the closed-loop
  Δhit-rate / Δpath-efficiency interpretable. This is the number that estimates
  real-trial benefit.
- **Precedent.** Lee et al. trained their copilot **entirely in simulation** and
  found it "generalized to increase the closed-loop BCI centre-out 8 task
  performance" (2.1–3.9× hit rate) in a participant with paralysis. The transfer
  path we are proposing is the one already demonstrated in the reference paper.

`closed_loop.py` implements legs 1–2 and a preliminary leg 3.

### The honest caveat (and its mitigation)

The surrogate models a **target-directed** user, not a **recorded reactive
human**: our EEGK data was collected without a copilot in the loop, so it does not
contain how a human *responds* to copilot perturbations. This is the one place the
argument can break. Mitigations, in order of strength: (a) leg-1/leg-2 validation
above; (b) sensitivity analysis over the user model (vary σ, correct-direction
fraction, reaction lag) and report the copilot gain across that range, not at a
single operating point; (c) the decisive test — a small **closed-loop human pilot**
— which is exactly what a positive surrogate result is meant to justify funding.

---

## 6. Status of the code

**Implemented and verified on the fixed loader:** the control law (`copilot_core.py`);
the leakage-free within-subject protocol (`copilot_dataset.split_real`, now blocking
by session folder); per-session loading and metric-safe cleaning
(`copilot_dataset.py`); the learning / feature / cleaning diagnostics (`studies/`).

**Implemented, not yet re-measured on the fixed loader:** the closed-loop surrogate
and its legs 1–2 (`closed_loop.py`), the reactive variant
(`experimental/closed_loop_reactive.py`), the blend machinery
(`blend_constructor.py`, `sim_scaling.py`), trajectory-aware steering
(`trajectory_aware_copilot.py`).

**Not implemented:** a `--clean` path through `train_copilot.py` /
`evaluate_copilot.py` / `closed_loop.py` — cleaning currently reaches the model only
through the `studies/` scripts, which call `load_source(clean=True)` themselves.

---

## 7. Which conclusions currently stand

| Conclusion | Status |
| --- | --- |
| Additive corrective velocity is the right control law | ✅ Stands |
| Within-subject, one model per subject | ✅ Stands |
| Open-loop copilot gain ≈ +1 pp (+1.43 pp cleaned) | ✅ Re-measured on the fixed loader, 3 seeds |
| The LSTM learns target information from trajectory alone | ✅ New (7/29), on the fixed loader |
| Cleaning helps; trimming is the active ingredient | ✅ New (7/29), on the fixed loader |
| Feature engineering is exhausted; signal is in velocity, not position | ✅ New (7/29), on the fixed loader |
| Smarter open-loop steering fails | ✅ Stands qualitatively; magnitudes pending re-run |
| Sim helps only as a ~25% minority augmenter | ⚠️ Pending re-run (corrupted trial set) |
| ~87% of decoder errors carry no recoverable intent | ⚠️ Pending re-run; and it never tested whether the model learns |
| The 7/22 arm-phase / language probes | ⚠️ Pending re-run |
| Raw-decoder baseline | ❌ Superseded: **64.0%**, not ~53–55% |

---

## 8. Next steps

**Blocking — re-establish the baselines.** Re-run, on the fixed loader, the
training-data composition study (`archive/sweep_blends.py`), the ceiling diagnostic
(`archive/diagnose_residuals.py`), and the 7/22 probes (`experimental/`). Until this
is done we do not know which of the last month's conclusions survive, and the
corrected 64.0% baseline changes the denominator of every headroom argument.

**Then, in priority order:**
1. Adopt cleaned inputs as the default path — trimming unconditionally, rescaling
   only on top of trimming. This means wiring a `--clean` flag through
   `train_copilot.py` / `evaluate_copilot.py` / `closed_loop.py`.
2. Harden `closed_loop.py`: multi-seed legs 1–2; confirm surrogate faithfulness
   holds per subject before trusting any closed-loop number.
3. Closed-loop A/B of instantaneous vs trajectory-aware steering — the setting
   where trajectory-aware is predicted to finally win.
4. User-model sensitivity sweep, including the persistence parameter α in
   `experimental/closed_loop_reactive.py` (the §5 caveat mitigation).
5. Key-prior conditioning: fold the LM direction-prior into the copilot input.
6. If closed-loop gains survive the sweeps, write up the human-pilot proposal.

**Deprioritized (with reasons):** feature-space tuning (closed 7/29 — every richer
set is within seed noise); surrogate as *training data* (target-independent
structure); pure-RL copilot (see `legacyRL/README_legacy.md` — reward hacking,
directional bias); higher sim fractions in the blend.

**Open question for discussion.** The target signal is carried by per-tick velocity,
while the task metric scores the accumulated endpoint position (7/29 §5). Is the
fixed-length endpoint metric the right objective for the arm phase at all?
