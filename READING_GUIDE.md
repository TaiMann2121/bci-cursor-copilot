# How to read this codebase

A staged walkthrough of the arm-BCI copilot pipeline, in dependency order. Roughly
**3–4 hours end to end**; stages 0–4 (~2 h) are enough to understand the product,
stages 5–7 are why it looks the way it does and where it goes next.

Every file has a module docstring stating the question it answers. **Read the
docstring first, then the code.** Line numbers below are anchors, not gospel — they
drift as the code changes.

---

## The one-paragraph version

A subject sits in front of 8 targets. EEGNet decodes their intent into a noisy
per-tick velocity; the cursor integrates it for 13 ticks (2 s); whichever of the 8
directions the final cursor position points at is the selected key. That is ~64%
accurate. The **copilot** is an LSTM that watches the trajectory tick by tick,
maintains a belief over the 8 targets, and each tick adds a small velocity pointing
from the current cursor toward its predicted target, scaled by its confidence. The
whole repo is: build that dataset cleanly, train that LSTM, and measure honestly
whether the added velocity helps.

---

## Data flow

```
data/OnlineArmTrajectoryEEGK/*/*/online_arm_trajectories.csv
        │
        │  copilot_dataset.load_source("eegk_real", clean=?)      ← per session folder
        ▼
   List[Trajectory]           .pos (T,2) normalized · .target_label · .session_id
        │
        ├── copilot_dataset.split_real(seed)     → per subject {train, val, test}
        │                                          TEST = whole held-out session blocks
        ▼
   copilot_dataset.build_features / copilot_core._tick_features
        │                                  [x, y, vx_unit, vy_unit, vel_mag_z]
        ▼
   copilot_core.LSTMCopilot  ── per-tick 8-way logits ──▶ softmax belief
        │
        │  copilot_core.corrective_velocity(cursor, pred, conf, vel_mag)
        ▼
   cursor_{t+1} = cursor_t + bci_vel_t + correction_t        ("additive" mode)
        │
        ▼
   copilot_core.angle_pred(final cursor)  →  accuracy vs raw-BCI accuracy = the gain
```

Two loops sit on top of this: **training** (`train_copilot.py`, DAgger — the model
re-simulates its own trajectories and trains on them) and **evaluation**
(`evaluate_copilot.py`, open-loop replay on held-out blocks). The frontier is
replacing open-loop replay with a **closed loop** (`closed_loop.py`), where a
surrogate user re-aims from wherever the copilot put the cursor.

---

## Stage 0 — Orientation (20 min, no code)

1. `README.md` — status and the map.
2. `PIPELINE.md` — the architecture argument. Note the status banner: the argument
   in §3–§5 stands, but §2's numbers were measured on a corrupted trial set, fixed
   on 7/29 (§2.5). §7 tells you which conclusions currently hold.
3. `Progress Reports/7_29 Progress Report.docx` — the most recent evidence. If you
   read only one report, read this one; it supersedes parts of the others.

**Hold these questions through everything below:** *What exactly is being measured?
Could the training data have leaked into it? Is the effect bigger than seed noise?*

---

## Stage 1 — The data foundation (45 min) · `copilot_dataset.py` (540 lines)

The most important file in the repo. Everything else assumes its vocabulary.

| Read | Line | Why it matters |
| --- | --- | --- |
| `Trajectory` | `copilot_dataset.py:83` | The atom. `pos` is (T, 2) in normalized units (target circle = radius 1); `keys` is the 5-tuple trial id; `session_id` is the field the 7/29 loader fix added. |
| `label_from_position`, `angle_error_deg`, `evaluate_final_positions` | `copilot_dataset.py:106` | **The metric.** Direction-only: argmax dot-product of the final cursor against the 8 unit directions. Distance from center is irrelevant — which is exactly why the cleaning transforms below cannot change a label. |
| `load_source` | `copilot_dataset.py:164` | The one entry point for data. Note `clean=False` is the default and read the docstring's NOTE about what that does and does not mean. |
| `_load_eegk_real_sessions` | `copilot_dataset.py:239` | **The 7/29 fix.** Real data is loaded per session folder and never concatenated, because trial keys collide across folders. The old behavior fused 3,030 pairs of different-target trials. |
| `trim_leading_dwell`, `clean_trajectory_sessions` | `copilot_dataset.py:264` | The cleaning: trim pre-onset dead ticks, then rescale each session to a common median endpoint radius. Both preserve direction ⇒ metric-safe. |
| `per_tick_velocity`, `compute_norm_stats`, `build_features` | `copilot_dataset.py:389` | The 5 features the model sees: `[x, y, vx_unit, vy_unit, vel_mag_z]`. Velocity is a finite difference of position — that is the whole "decoder output". |
| `split_real` | `copilot_dataset.py:465` | **The leakage story.** Test is whole held-out `(session_folder, session, run)` blocks, not random trials, so trials from the same block never straddle train/test. |

**Run it:**

```bash
python -c "import copilot_dataset as cd; r=cd.load_source('eegk_real'); c=cd.load_source('eegk_real', clean=True); print(len(r), cd.baseline_metrics(r)); print(len(c), cd.baseline_metrics(c))"
```

You should see 16,197 trials both times and **identical** accuracy (64.04%) — that
identity *is* the metric-safety proof.

**You can now answer:** What is one trial? What counts as correct? Where could
leakage enter, and what stops it?

---

## Stage 2 — The model and the control law (45 min) · `copilot_core.py` (321 lines)

The single definition of what the copilot *does*. Small file, highest density.

| Read | Line | Why it matters |
| --- | --- | --- |
| `LSTMCopilot` | `copilot_core.py:38` | 2-layer LSTM, hidden 64, per-tick 8-way head. Deliberately small — the constraint is data, not capacity. |
| `resolve_vel_mag` | `copilot_core.py:61` | `"inv_ticks"` (1/T) vs a fixed float (production uses `0.02`). This scalar is the copilot's *authority*; note it is **absolute** while BCI velocity scales with the session — the confound `studies/ablate_cleaning.py` chases. |
| `corrective_velocity` | `copilot_core.py:70` | **The control law, 8 lines.** Unit vector from current cursor toward the predicted target's circle position, times `vel_mag`, times the softmax confidence. Confidence-scaling is what makes an unsure copilot harmless. |
| `angle_pred` | `copilot_core.py:87` | The same argmax metric as Stage 1, applied to the copilot-moved cursor. |
| `_tick_features` | `copilot_core.py:98` | Must match `build_features` exactly — the model is fed the same 5 numbers whether it is training or rolling out. |
| `simulate_trajectory` | `copilot_core.py:118` | **The heart of the repo.** One tick: featurize → LSTM step → belief → correction → `cursor += bci_vel + correction`. Read the `velocity_mode` docstring: in `additive`, a zero-correction run reproduces the recorded BCI endpoint exactly, which is what makes the copilot-vs-BCI comparison a strict A/B. |
| `make_tick_weights`, `masked_weighted_ce` | `copilot_core.py:202` | Loss weighting over ticks (`exponential`, exponent 3) — late ticks matter more, because the endpoint is what is scored. |
| `simulate_batch` | `copilot_core.py:236` | Vectorized `simulate_trajectory` for DAgger (the 120× speedup). Same math; read it only after the scalar version is clear. |

**You can now answer:** Given a cursor at (0.3, 0.1) and a belief of 0.6 on
direction N, what velocity does the copilot add this tick? Why does confidence
multiply the correction?

---

## Stage 3 — Training (30 min) · `train_copilot.py` (299 lines)

| Read | Line | Why it matters |
| --- | --- | --- |
| Module docstring | `train_copilot.py:1` | `within_subject` = one model per subject (a pooled model cannot capture subject-specific tendencies). |
| `trial_view`, `SeqDataset` | `train_copilot.py:51` | Trial → `(bci_vel, raw_pos, label)`; padding + mask. |
| `train_one_model` | `train_copilot.py:94` | **The two-phase loop.** Phase 1 (3 epochs): train on the *recorded* trajectory features. Phase 2 (25 epochs): **DAgger** — re-simulate with the current model in the loop (`core.simulate_batch`) and train on the distribution the copilot itself induces. This is the standard fix for train/deploy distribution mismatch: a copilot trained only on recorded paths never sees the paths it creates. |
| Best-epoch selection | `train_copilot.py:146` | Checkpoints on validation **copilot accuracy**. Worth noting when reading the 7/29 report's critique of accuracy-based selection. |
| `load_training_trajectories`, `group_trajectories`, `main` | `train_copilot.py:164` | Source selection, per-subject grouping, and the run manifest written to `runs/`. |

**You can now answer:** Why two phases? What does the model see in phase 2 that it
never sees in phase 1?

---

## Stage 4 — Evaluation (25 min) · `evaluate_copilot.py` (239 lines)

The authoritative number. Read `evaluate` (`evaluate_copilot.py:72`) and pay
attention to two flags: `--eval_split test` (held-out blocks; `all` is the legacy
leaky behavior, kept only for reproducing old runs) and `--split_seed`, which
**must** match the seed used to build the training data or the split silently
differs.

The reported quantity is always a **paired difference on the same trials**: raw-BCI
endpoint accuracy vs copilot endpoint accuracy. In `additive` mode the copilot run
with zero correction is exactly the BCI run, so the difference is attributable to
the correction alone.

**You can now answer:** What would make a reported gain not real? (Answers: wrong
split seed, `--eval_split all`, single seed, comparing across different trial sets.)

---

## Stage 5 — Why the pipeline looks like this (60 min) · `studies/`

Read `studies/README.md`, then the four scripts in order. Each is one question and
reuses the Stage 1–4 modules rather than reimplementing them, so they double as
worked examples of the API.

1. **`studies/diagnose_learning.py`** — *Is the model sound at all?* Strips the
   control law away and trains the LSTM as a plain 8-way classifier, logging
   train/val cross-entropy against chance (ln 8 = 2.079). Answer: yes — train CE
   falls far below chance for all six subjects, validation tracks it. This is the
   gating result; without it, "the copilot only gains 1 pp" is uninterpretable.
2. **`studies/feature_sweep.py`** — *Which features carry the signal?* Position-only
   is ~6σ worse than anything containing velocity; velocity-only matches the full
   set. **The signal is in per-tick velocity, while the metric scores accumulated
   position** — that tension is the sharpest open question in the project.
3. **`studies/experiment_clean.py`** — *Does cleaning move the end-to-end gain?*
   +1.10 → +1.43 pp, variance halved. Note the design: identical control law,
   identical trials, only the inputs differ.
4. **`studies/ablate_cleaning.py`** — *Trim or rescale?* Trimming does the work;
   rescaling alone is slightly harmful; they interact. Also read Q2 in its
   docstring — a genuinely subtle confound (an absolute `vel_mag` against a
   session-scaled BCI velocity) found on review and measured rather than argued.

**You can now answer:** Why is the current feature set not the bottleneck? Why is
"trim, but do not rescale without trimming" the guidance?

---

## Stage 6 — The sim/blend branch (30 min, optional) · `sim_scaling.py`, `blend_constructor.py`

Context for the training-data composition study — a closed line of work whose
numbers are pending re-run on the fixed loader. Read `sim_scaling.calibrate_sim`
(`sim_scaling.py:285`) for how simulated trajectories are mapped onto a subject's
real radius and onset dwell, and `blend_constructor.build_blend`
(`blend_constructor.py:113`) for fixed-budget real:sim mixing. Skip on a first pass
if you are short on time; nothing downstream of Stage 5 depends on it.

Related characterization, also optional: `analysis/profile_sources.py` produces the
per-subject statistics these calibrations are fit to.

---

## Stage 7 — The frontier (45 min)

| Read | What to take from it |
| --- | --- |
| `PIPELINE.md` §3–§5 (re-read) | The argument that **open-loop replay is the bottleneck**: a recorded velocity stream cannot react to the copilot, so a copilot's feedback benefit is structurally unmeasurable open-loop. |
| `closed_loop.py` — `SubjectSurrogate` (`:85`), `_rollout` (`:135`), `calibrate_sigma` (`:206`), `main` (`:282`) | The surrogate: heading = angle(target − **current cursor**) + noise, magnitude from the subject's real step texture, σ calibrated so the surrogate reproduces that subject's real accuracy open-loop. Because it re-aims from the current cursor, the copilot's displacement changes the next command. `main` runs the 3 validation legs. |
| `trajectory_aware_copilot.py` — `simulate_batch_traj_aware` (`:49`), `evaluate_ab` (`:131`) | Accumulated-belief steering. It **loses** open-loop (−0.18 to −2.26 pp); the hypothesis is that decisive early steering only pays when the loop is closed. A clean example of a negative result kept because it is a prediction, not a failure. |
| `experimental/closed_loop_reactive.py` | The more honest surrogate: a data-driven one plus an explicit **user-persistence α** sweeping open-loop (α=0) to fully closed-loop (α=1). The α-dependence *is* the result. |
| `experimental/README.md`, `word_decode.py`, `sentence_decode.py`, `sentence_rerank.py` | Problem 2: language-model word/sentence decoding, and the bridge back to Problem 1 (a language prior over the *next direction*). |

**You can now answer:** What would a closed-loop gain prove, and what would it not?
(See `PIPELINE.md` §5 — the surrogate models a target-directed user, not a recorded
reactive human.)

---

## Stage 8 — Skim only

- `archive/` — retired but runnable: the endpoint-modeled surrogate, the blend-ratio
  sweep, the correctability-ceiling diagnostic. Read `archive/README.md` for why each
  was retired; the last two are the scripts to re-run on the fixed loader.
- `legacyRL/` — the Lee et al. PPO infrastructure. Read `legacyRL/README_legacy.md`
  (the Run3–Run7 table, ending in reward hacking) and, if you are working on
  `closed_loop.py`, skim `legacyRL/SJtools/copilot/env.py` as prior art.
- `legacySL/` — the pre-refactor supervised scripts and the V3 model.

---

## If you have 30 minutes, not 4 hours

`README.md` status → `copilot_core.corrective_velocity` and `simulate_trajectory` →
`copilot_dataset.split_real` → `PIPELINE.md` §3. That is the control law, the
honesty guarantee, and the open problem.

## Suggested order for a supervisor review

`README.md` → `PIPELINE.md` §2.5 and §7 (what the loader defect invalidated, what
survives) → `studies/README.md` (the four 7/29 answers in one table) → the 7/29
report. Code only where a claim needs checking.
