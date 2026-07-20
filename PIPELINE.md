# The Copilot Pipeline: architecture decision and the credibility argument

This document states the final pipeline for the arm-BCI copilot and the argument
for why its improvements are likely to carry over to real human trials. It is the
one file to read first. Code details live in the module docstrings; the empirical
history lives in `Progress Reports/`.

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

1. **The control law works, but the open-loop gain is small.** A within-subject
   copilot improves endpoint accuracy by roughly **+1 pp** over the raw decoder,
   reproduced on the current 6-subject data (raw 53.3% → copilot 54.3%, seed 0).

2. **Training-data composition has been mined out.** Real-only is a strong
   baseline; calibrated EEGK-sim helps only as a **~25% minority augmenter**
   (+0.9 pp, leakage-free, 3 seeds); higher sim fractions and pure sim are worse;
   the **endpoint-modeled surrogate does not help as training data** because it
   carries target-*independent* structure the copilot cannot exploit.

3. **Open-loop correctability is capped, and the cap is not a data problem.** The
   residual diagnostic showed ~87% of the decoder's wrong trials carry *no
   recoverable intent* in the replayed trajectory. Sharper still: the model's
   belief points at the true target on ~28% of wrong trials, but only ~8%
   converts into the endpoint — and every attempt to convert that gap with
   smarter **open-loop** steering (`trajectory_aware_copilot.py`) **fails**:
   causal accumulated-belief steering never beats the instantaneous copilot, and
   steering harder makes it monotonically worse (−0.18 to −2.26 pp).

**These three facts point at one conclusion, and it is the crux of this document.**

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

So the ~87% "uncorrectable" ceiling and the belief→endpoint gap are **artifacts of
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

## 6. Status and next steps

**Done / settled:** control law; leakage-free within-subject protocol; training-data
composition study; open-loop ceiling diagnosis; closed-loop surrogate scaffold with
legs 1–2 running.

**Next, in priority order:**
1. Harden `closed_loop.py`: multi-seed legs 1–2; confirm surrogate faithfulness
   holds per subject before trusting any closed-loop number.
2. Closed-loop A/B of instantaneous vs trajectory-aware steering — the setting
   where trajectory-aware is predicted to finally win.
3. User-model sensitivity sweep (the caveat mitigation).
4. Key-prior conditioning: fold the LM direction-prior into the copilot input.
5. If closed-loop gains survive the sweeps, write up the human-pilot proposal.

**Deprioritized (with reasons):** surrogate as *training data* (target-independent
structure; ceiling is open-loop, not data); pure-RL copilot (see
`legacyRL/README_legacy.md` — reward hacking, directional bias); higher sim
fractions in the blend.
