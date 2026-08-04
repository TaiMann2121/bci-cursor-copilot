# The Copilot Pipeline: architecture decision and the credibility argument

This document states the final pipeline for the arm-BCI copilot and the argument
for why its improvements are likely to carry over to real human trials. Read it
after `README.md`; for a code-reading order see `READING_GUIDE.md`. Code details
live in the module docstrings; the empirical history lives in `Progress Reports/`.

> **Status of this document (updated 8/4/2026).** Two rounds of correction have
> passed over this file. (1) A loader defect (§2.5) corrupted the real-data trial
> set behind every pre-7/29 number; the pooled raw-decoder baseline is **64.04%**,
> not the ~53–55% quoted below and in earlier reports. (2) On 8/4 three further
> defects surfaced (§2.6): checkpoint selection was saving mode-collapsed
> classifiers, test blocks were selected deterministically so no result had ever
> been tested across splits, and torch was unseeded so runs were irreproducible.
>
> **What survived all of it:** the copilot's gain is real and generalizes —
> **+1.01 ± 0.39 pp over a shuffled-label null, 8/8 randomized splits**.
> **What did not:** the diagnosis in §3 that open-loop *evaluation* is the
> bottleneck. See §2.6 and §7.
>
> §2 marks each claim by whether it survives; §7 sorts the standing conclusions.
> **§8 is superseded by [`ROADMAP.md`](ROADMAP.md).** Do not quote a number from
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

1. **The control law works, the open-loop gain is small, and it now has a null
   control.** ✅ *Survives, re-measured, and null-controlled (8/4).* A
   within-subject copilot improves endpoint accuracy by roughly **+1 pp** over the
   raw decoder. On the fixed loader and held-out test blocks (3 seeds): raw inputs
   **+1.10 ± 0.30 pp**, cleaned inputs **+1.43 ± 0.16 pp**.
   The 8/4 null control (§2.6) puts this on firm ground: across **8 randomized
   splits**, trained **+1.30 ± 0.28 pp** vs a **shuffled-label null at
   +0.30 ± 0.30 pp**, paired difference **+1.01 ± 0.39 pp, positive on 8/8**
   (t = 7.2, CI [+0.73, +1.28]). A maximally confident but permanently *wrong*
   pusher loses **6.75 pp**, so the control law is not flattering the metric.
   **The honest headline is the null-subtracted +1.01 pp.**

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
   converting into the endpoint. Three caveats now attach: it was measured on the
   corrupted trial set; it predates the 8/4 selection fix, so the model behind it
   may have been mode-collapsed; and it never measured whether the classifier
   *learns*. What remains solid is that smarter **open-loop** steering
   (`trajectory_aware_copilot.py`) **fails**: causal accumulated-belief steering
   never beats the instantaneous copilot, and steering harder makes it
   monotonically worse (−0.18 to −2.26 pp).

4. **The classifier reads the trajectory about as well as the endpoint does.**
   ✅ *New (8/4), one split.* With checkpoint selection fixed, the direct
   classifier readout equals the raw endpoint argmax — **65.1% vs 65.1%** on
   matched held-out trials — and on trials where the two disagree neither
   dominates (37% vs 42%). Note this is **not** a demonstrated ceiling: it is
   equally consistent with "the trajectory carries nothing beyond the endpoint"
   and with "the classifier is fine now that it trains properly." The current data
   does not separate those.
   *Superseded reading:* the 7/29 learning diagnostic's 57–78% final-tick accuracy
   comes from a **stratified random** split, where trials from one session appear
   in both train and val. It measures *fitting*, not cross-session generalization,
   and must not be compared against session-blocked numbers.

**These facts point at one conclusion, and it is the crux of this document.** The
re-runs in `ROADMAP.md` can change the magnitudes; the structural argument in §3
does not depend on them — but see §2.6, which undercuts §3's *diagnosis*.

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

## 2.6. The three defects found on 8/4, and what they cost

Found while running the §8 re-runs (`experimental/readout_probe.py`,
`studies/null_control.py`). All three are fixed; each invalidated a class of
earlier measurement.

1. **Checkpoint selection was saving mode-collapsed classifiers.**
   `train_copilot.train_one_model` selected its checkpoint on validation *copilot
   endpoint* accuracy while training on per-tick CE. Those are decoupled: the
   correction is small, so a classifier that had collapsed onto the single most
   frequent label still posts a passable endpoint and gets saved as "best". This
   happened in **3 of 4 training configs** — S04's saved model predicted one
   constant class on every tick (readout 18.5%, exactly its most-common test
   label; S03 exactly 19.8%). Fixed: `--select_on val_ce` is now the default, and
   `evaluate_group` reports `ce`/`cls_acc` so a collapse is visible during
   training. After the fix S03 19.8% → 63.1%, S04 18.5% → 77.1%.
   **Cost:** any trajectory-side conclusion measured before 8/4 may have been
   measured on a collapsed model.

2. **Test blocks were selected deterministically.** `split_real` sorted blocks by
   size and took them smallest-first; the rng touched only the val split. Every
   "seed" therefore shared one identical held-out set, and **no result in this
   project's history had ever been tested across splits.** Fixed:
   `random_test_blocks=True` is the default (`False` reproduces pre-8/4 splits).
   **Cost:** held-out baselines turn out to range **63.3–66.6%** depending on which
   sessions are held out — so every "how much headroom is left" argument computed
   on one split carries ±1.4 pp it never accounted for. (The pooled 64.04% over
   all 16,197 trials is unaffected; it is a different statistic.)

3. **Torch was never seeded on the `train_copilots` path.** `train_copilot.py`
   seeds inside `main()`, which `closed_loop.train_copilots` bypasses. Model init
   came from OS entropy, so nominally identical runs differed: seed 0 measured
   +0.85 pp in one process and +1.60 pp in another. Fixed: seeded globally and
   per subject.
   **Cost:** single-run numbers in earlier reports carry ~±0.4 pp of unlabelled
   init noise.

**The methodological lesson, which is why the null control exists.** A single
split or single training run pointed the *wrong way* twice in one day here: on one
split the copilot read +0.85 pp against a +0.28 pp null and looked like noise;
across 8 splits the null centres near zero and the effect is solid at
+1.01 ± 0.39 pp. Report gains against a null and across splits, or do not report
them.

---

## 3. The diagnosis: the open-loop *evaluation* is the bottleneck

> ⚠️ **Weakened 8/4.** This section argues the copilot's belief is fine and the
> *open-loop replay* is what caps the gain. Two 8/4 results cut against it. The
> ~87% "uncorrectable" figure it leans on came from a run predating the selection
> fix, so it may describe a collapsed model rather than a property of the data
> (§2.6). And with selection fixed, the classifier readout only *matches* the raw
> endpoint (§2 item 4) — a belief at parity with the trivial readout is not a
> belief being held back by the evaluation protocol. The closed-loop pivot may
> still be right, but this diagnosis is no longer the argument for it; the
> compounding effect across characters in the language architecture is
> (`ROADMAP.md` Phase 4).

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
by session folder **and randomizing which blocks are held out**); per-session loading
and metric-safe cleaning (`copilot_dataset.py`); the learning / feature / cleaning
diagnostics (`studies/`); the null control (`studies/null_control.py` +
`aggregate_null_control.py`) and the readout probe
(`experimental/readout_probe.py`, now with `--clean` / `--sim_frac` / `--eval_norm`).

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
| **The copilot gain is real and generalizes: +1.01 ± 0.39 pp over a shuffled-label null, 8/8 randomized splits** | ✅ **New (8/4), null-controlled** |
| The control law is not free lunch (a confident wrong pusher loses 6.75 pp) | ✅ New (8/4) |
| Cleaning helps; trimming is the active ingredient | ✅ New (7/29), on the fixed loader |
| Feature engineering is exhausted; signal is in velocity, not position | ⚠️ Predates the 8/4 selection fix — re-run before quoting |
| Smarter open-loop steering fails | ✅ Stands qualitatively; magnitudes pending re-run |
| Classifier readout ≈ raw endpoint (65.1% vs 65.1%) | ✅ New (8/4), but **one split**, and not a demonstrated ceiling |
| The LSTM learns target information from trajectory alone | ⚠️ Qualified: the 57–78% is from a *random* split (measures fitting, not cross-session generalization) |
| Sim helps only as a ~25% minority augmenter | ⚠️ Pending re-run; and sim in the blend is one of the conditions that *triggered* mode collapse (§2.6) |
| ~87% of decoder errors carry no recoverable intent | ⚠️ Pending re-run; may describe a collapsed model |
| The 7/22 arm-phase / language probes | ⚠️ Pending re-run |
| Open-loop *evaluation* is the bottleneck (§3) | ⚠️ Weakened (8/4) — see §3 banner |
| Pooled raw-decoder baseline (all 16,197 trials) | ✅ **64.04%**, not ~53–55% |
| A single held-out baseline figure | ❌ Superseded: held-out baselines range **63.3–66.6%** by split (§2.6) |

---

## 8. Next steps

> **Superseded 8/3/2026 by [`ROADMAP.md`](ROADMAP.md).** The 8/3 supervisor
> meeting green-lit the pivot to a joint language + trajectory copilot, which
> reprioritizes the list below (notably: the blend re-run is deliberately
> dropped, and the language probes move onto the critical path). The section is
> kept for the reasoning; follow `ROADMAP.md` for what to actually do next.

**Blocking — re-establish the baselines.** Re-run, on the fixed loader, the
training-data composition study (`archive/sweep_blends.py`), the ceiling diagnostic
(`archive/diagnose_residuals.py`), and the 7/22 probes (`experimental/`). Until this
is done we do not know which of the last month's conclusions survive, and the
baseline changes the denominator of every headroom argument. *(8/4: partly done —
`readout_probe` and the null control are complete; the blend re-run is deliberately
dropped. The denominator point is sharper than written here: held-out baselines
range 63.3–66.6% by split, so there is no single denominator — see §2.6.)*

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
