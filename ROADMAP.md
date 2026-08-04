# Roadmap (from 8/3/2026)

Supersedes `PIPELINE.md` §8. Written after the 8/3 supervisor meeting, which
green-lit the pivot from **trajectory-only** to **joint language + trajectory**
copilot.

**The one-line plan:** spend the wait for her materials firming up the argument
for the pivot (Phase 0), then build the copilot around the constrained-vocabulary
language prior (Phases 2–3), evaluating everything on post-autocorrect word
accuracy (Phase 1).

---

## The argument we are making now

The old framing — "the model hit a ceiling near the raw baseline" — is not
defensible on its own; loss decreasing proves learning, not a ceiling.

An earlier draft of this file replaced it with "a trajectory-reading LSTM gets
~66% vs the endpoint's 64.0%, so the trajectory carries no extra information."
**That is withdrawn.** The 66% came from `diagnose_learning`, which uses a
*stratified random* within-subject split; the production protocol holds out whole
(session, run) blocks. The two are not comparable, and the matched measurement
(8/4, task 0.1) is 60.4% classifier vs 62.7% endpoint on identical held-out
trials.

**What the 8/4 runs actually established** (`results/readout_probe_seed0_*.log`,
`results/null_control_seed0_clean_realonly.log`):

1. **Cleaning is metric-safe — verified.** The raw-endpoint column is 62.7% in
   all eight runs, unchanged by trimming/rescaling. Confirms PIPELINE §2.5.
2. **Training was collapsing in 3 of 4 configs — now fixed (0.1d).** With sim in
   the blend, or with raw inputs, the classifier mode-collapsed onto the single
   most frequent label (S04 read exactly 18.5% = its most-common test label; S03
   exactly 19.8%). Cause: `train_one_model` selected its checkpoint on validation
   *copilot endpoint* accuracy while training on per-tick CE, so a collapsed
   classifier could still be saved as "best". Selecting on `val_ce` removes it —
   S03 19.8% → 63.1%, S04 18.5% → 77.1%, no subject collapsed.
   **Consequence:** with selection fixed, the classifier readout **matches the
   raw endpoint** (65.1% vs 65.1% aggregate) and the disagreement split moves to
   near-parity (37% vs 42%, with the classifier winning on S01 and S03). The
   earlier reading that "the classifier is meaningfully worse than the endpoint"
   was an artifact of the broken selection. Caveat: that comparison is on one
   randomized split, and is not directly comparable to the pre-fix runs, which
   used the old deterministic split.
3. **The control law is not free lunch.** A maximally confident but permanently
   wrong pusher loses **6.72 pp** (best fixed class still loses 1.9). So the
   corrective velocity does not mechanically flatter the endpoint metric.
4. **The copilot gain is real and it generalizes across sessions.** Over **8
   randomized splits** (0.1e): trained **+1.30 ± 0.28 pp** vs a shuffled-label
   null at **+0.30 ± 0.30 pp**; paired within-split **+1.01 ± 0.39 pp, positive
   in 8/8, t = 7.2, ~95% CI [+0.73, +1.28]**. A confidently-wrong constant pusher
   loses **6.75 ± 0.61 pp**, so the control law is not free lunch. Target
   inference is genuinely contributing, and the null-subtracted **+1.01 pp** is
   the honest headline — close to the +1 pp that has been on record all along.
   *A single seed was misleading here*: seed 0 alone read +0.85 trained vs +0.28
   null and looked like noise. Never call this from one run.
   Remaining caveat: confidence is unmatched (trained 0.481 vs shuffled 0.163),
   so trained applies ~3x more correction and part of the margin is nudge
   strength rather than better aim (task 0.1h).

**Consequence for the pivot.** The copilot itself is sound and the +1 pp result
survives scrutiny — that part of the last month holds up. What does *not* hold is
the claim that the trajectory path is mined out: most measurements behind it came
from configs where training had mode-collapsed (finding 2). So the pivot to
language remains the right call (supervisor green-lit it, and the language axis
has far more headroom), but argue it **on the size of the language lever, not on
a trajectory ceiling we have not demonstrated.**

---

## Phase 0 — Firm up the argument *(now; not blocked on anything)*

Small and cheap. Do this while waiting for her materials.

| # | Task | Why it matters | Est. | Status |
| --- | --- | --- | --- | --- |
| **0.1** | Run `experimental/readout_probe.py` on the fixed loader | Matched raw-vs-classifier comparison on identical held-out trials. | ~half day | ✅ **done 8/4** — see findings above. Gained `--clean`, `--sim_frac`, `--eval_norm` flags; `train_copilots` gained `sim_frac`. |
| **0.1b** | Null-classifier control (`studies/null_control.py`) | The control the +1 pp result never had. Refuted the control-law-artifact hypothesis; showed the effect is inside single-seed noise. | ~half day | ✅ **done 8/4** |
| **0.1c** | Multi-seed the healthy cell (5 runs) + `studies/aggregate_null_control.py` | Resolved it: paired trained − shuffled = **+1.39 ± 0.31 pp, 5/5 runs**. The copilot gain clears its null. | ~1–2 h | ✅ **done 8/4** |
| **0.1d** | Fix checkpoint selection in `train_copilot.train_one_model` | Was selecting on val copilot-endpoint while training on per-tick CE, letting a mode-collapsed classifier be saved as "best". Now `select_on="val_ce"` by default (`cls_acc`/`copilot` also available); `evaluate_group` returns `ce` + `cls_acc` and the epoch log prints both. | ~half day | ✅ **done 8/4** — collapse gone: S03 19.8%→63.1%, S04 18.5%→77.1% |
| **0.1e** | Randomize test-block selection in `split_real` | `random_test_blocks=True` is now the default; blocks are added only if they fit under `test_frac` (the old code took the first block unconditionally, unsafe once order is shuffled). `False` reproduces pre-8/4 splits exactly. | ~half day | ✅ **done 8/4** — exposed a 63.3–66.6% baseline spread |
| **0.1f** | Seed torch in `train_copilots` | `train_copilot.py` seeds torch inside `main()`, which `train_copilots` bypassed. Now seeded globally and per subject (`seed*997 + i`). | ~15 min | ✅ **done 8/4** — identical seeds give identical weights |
| **0.1g** | Re-run 0.1c across randomized splits (8 splits) | **The gain generalizes.** Paired trained − shuffled = **+1.01 ± 0.39 pp, 8/8 splits positive**, t = 7.2, CI [+0.73, +1.28]. | ~1–2 h | ✅ **done 8/4** |
| **0.1h** | Confidence-matched null | Last remaining confound: trained confidence 0.481 vs shuffled 0.163, so trained applies ~3x more correction and part of the margin is nudge strength, not aim. Close it by matching mean correction magnitude across arms (scale `vel_mag` inversely to mean confidence, or temperature-match the softmax). | ~half day | ⬜ **optional** |
| **0.2** | Re-run `experimental/prior_fusion_probe.py` on the fixed loader | Direction-level language prior × motor posterior. No longer a side probe — it is a prototype of the new architecture. Its numbers set expectations for Phase 2. | ~half day | ⬜ |
| **0.3** | Write down the deferral of the blend/composition study | Tests augmentation for a trajectory-only model we are stepping away from. Note that 0.1 found sim in the blend *causes* mode collapse — which is itself an argument against the 25% augmenter. | ~15 min | ⬜ |

**Done when:** 0.1c gives a multi-seed trained-vs-null comparison. Until that
lands, do not quote +1 pp (or any copilot gain) to the supervisor as established.

---

## Phase 1 — Evaluation design *(blocked on supervisor materials)*

The single most important design decision, and we cannot make it without her
autocorrect script.

**Why:** her autocorrect runs *after* the typed string exists. Our copilot runs
*during* the arm phase. If we measure arm-phase endpoint accuracy, we can show a
gain that autocorrect would have recovered anyway — and the result evaporates on
composition. **The metric must be post-autocorrect word/sentence accuracy.**

| # | Task | Blocked on | Est. |
| --- | --- | --- | --- |
| **1.1** | Obtain autocorrect script + word library | **Her** — ask for these two first | — |
| **1.2** | Stand up the composed pipeline: decoder → (copilot) → autocorrect → word | 1.1 | ~1 day |
| **1.3** | Measure the baseline to beat: raw decoder → autocorrect, no copilot | 1.2 | ~half day |
| **1.4** | Obtain finger-phase data + online word/sentence recordings | **Her** — needed for the character model and the real test set, but do not block 1.2–1.3 | — |

**Ask her for the autocorrect script and word library first.** The finger data
and recordings are needed for the model and test set; the script and library are
needed to design the evaluation at all.

---

## Phase 2 — The constrained-vocabulary prior *(the big lever)*

The user is constrained to a **closed word library** and the **word length is
known**. This is under-exploited and is the largest lever available.

- Known length largely dissolves the segmentation problem that
  `experimental/sentence_decode.py` identifies as the hard part (space shares
  group 5 with b/n).
- A closed library means the prior over the next direction is not a generic
  English character 4-gram (what the probes use today) but a **trie over library
  words still consistent with what has been typed**. That collapses toward
  near-deterministic after 2–3 characters.
- Because the trie prior is fixed for the whole arm phase of a given character,
  it can condition the model **from tick 0** rather than arriving as a post-hoc
  reweight.

| # | Task | Est. |
| --- | --- | --- |
| **2.1** | Build the trie over the word library; given (prefix, known length) emit the viable next-character set | ~1 day |
| **2.2** | Convert that to a prior over the 8 arm directions (marginalize characters → groups) | ~half day |
| **2.3** | Measure it standalone against the n-gram prior from 0.2 — same fusion, better prior | ~1 day |
| **2.4** | **Failure-mode handling**: out-of-library words, intended backspaces, and accumulated errors collapsing the trie to an empty set. Needs a confidence-gated prior weight and/or trie reset. | ~1 day |

**2.4 is not optional.** A confidently-wrong prior actively fights the user. It
also affects paradigm design, so raise it with her early rather than discovering
it in evaluation.

---

## Phase 3 — Joint model

| # | Task | Notes | Est. |
| --- | --- | --- | --- |
| **3.1** | Circular regression head (von-Mises or (sin, cos)) | Her suggestion, reframed. Won't move accuracy — same signal — but yields a **calibrated continuous posterior** that multiplies cleanly against the language prior, and the concentration parameter gives per-trial confidence for free. Worth ~a day as infrastructure, not as a standalone experiment. **Do not regress unsigned angular distance** — it discards correction direction and wraps badly at ±π. | ~1 day |
| **3.2** | Condition the LSTM on the Phase-2 prior from tick 0 | The actual joint architecture | ~2–3 days |
| **3.3** | Express the fused belief as corrective velocity via the existing control law | `copilot_core.py` is unchanged — see the architecture note below | ~half day |

### Architecture note (settles the endpoint-metric disagreement)

Her point about endpoint classification being the right *user feedback* is a UX
constraint, not a constraint on the internal representation. The resolution:

> **Endpoint stays the interface, not the objective.** Language prior and
> velocity signal fuse into the belief; the belief expresses itself as corrective
> velocity; the endpoint remains the readout. Nothing user-facing changes and no
> metric changes.

This closes the open question at the bottom of `PIPELINE.md` §8.

---

## Phase 4 — Closed loop

Her framing was "won't improve accuracy, but better simulates humans." There is a
stronger argument in the new architecture: **a correct character sharpens the
trie prior for the next character, so benefit compounds across characters.**
Open-loop replay structurally cannot show a compounding effect.

| # | Task | Est. |
| --- | --- | --- |
| **4.1** | Re-validate `closed_loop.py` legs 1–2 on the fixed loader (cheap; currently unmeasured) | ~1 day |
| **4.2** | Treat closed-loop as infrastructure for Phases 2–3, not as a result generator, until 4.1 passes | — |
| **4.3** | Closed-loop measurement of the compounding effect across characters | after Phase 3 |

**Sequencing risk:** `closed_loop.py` has not been re-measured since the loader
fix and its faithfulness legs are unvalidated. Do not generate headline numbers
from it before 4.1.

---

## Explicitly not doing (and why)

| Item | Reason |
| --- | --- |
| Blend / training-data composition re-run | Tests augmentation for a trajectory-only model we are stepping away from; already deprioritized in PIPELINE |
| Full residual ceiling diagnostic | Superseded by the learning diagnostic + task 0.1 |
| Angular *distance* regression as specified | Unsigned distance discards correction direction and wraps at ±π — use 3.1 instead |
| Further trajectory-only feature engineering | Closed 7/29; every richer set is within seed noise |
| Surrogate as training data | Carries target-independent structure the copilot cannot exploit |

---

## What to raise with her

1. **The 8/4 findings, in this order** — (a) **good news first**: the copilot
   gain passes a proper null control, +1.39 ± 0.31 pp over a shuffled-label
   model across 5 runs, so the headline result of the last month holds up and is
   if anything understated; (b) but training mode-collapses in 3 of 4 configs
   (`clean + real-only` is the only healthy one), so many prior trajectory
   measurements were made on broken runs; (c) therefore we are *not* claiming a
   trajectory ceiling — the pivot rests on the size of the language lever
   instead; (d) one caveat to state plainly: all of this sits on a single
   deterministic held-out set (task 0.1e).
2. **The metric has to be post-autocorrect** — otherwise copilot gains and
   autocorrect gains double-count. This is why the autocorrect script is the
   blocking material.
3. **Endpoint as interface, not objective** — the resolution to the 8/3
   disagreement; nothing user-facing changes.
4. **We are skipping the blend re-run** and why.
5. **The prior's failure modes (2.4)** — out-of-library words and backspaces
   affect paradigm design, so this needs her input early.

---

## Estimates

All estimates are rough and assume no surprises. Phase 0 is ~1–2 days total.
Phases 2–3 are the real work at ~1–1.5 weeks combined. Phase 1 is fast once
unblocked, but **nothing in Phases 2–4 is meaningfully measurable until Phase 1
lands**, so chase the autocorrect script and word library early.
