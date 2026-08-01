# arm-bci-copilot

An AI **copilot** for the arm phase of an EEG-BCI typing system. In the two-stage
paradigm, EEGNet decodes noisy per-tick cursor velocities and the final cursor
position selects one of 8 directions; the copilot observes the decoded trajectory,
infers the intended target, and adds a corrective velocity to aid the cursor.

## Where to start

| If you want to… | Read |
| --- | --- |
| Read the code in a sensible order | **[`READING_GUIDE.md`](READING_GUIDE.md)** — a staged walkthrough with file:line anchors |
| Understand *why* the pipeline looks like this | **[`PIPELINE.md`](PIPELINE.md)** — architecture decision + the transfer-credibility argument |
| See the empirical history | `Progress Reports/` — one .docx per week; **7/29 is the current one** |
| Understand the data on disk | `data/README_online_arm_trajectories.md` |

## Status (as of the 7/29/2026 report)

- **Control law is settled:** supervised LSTM target classifier → additive
  corrective velocity (`copilot_core.py`).
- **A loader defect corrupted every real-data number before 7/29 — now fixed.**
  Trial keys are not unique across session folders, so 3,030 pairs of distinct
  trials (different targets) had been merged. Real data is now loaded per session
  folder and split by session block. **Corrected raw-decoder baseline: 64.0%**
  over all 16,197 trials (not the ~53–55% quoted in earlier reports).
- **The model is functionally sound.** Trained as a plain 8-way classifier, per-tick
  training CE falls well below chance (ln 8 = 2.079) for all six subjects with
  validation tracking it; final-tick target accuracy 57–78% vs 12.5% chance
  (`studies/diagnose_learning.py`). There *is* decodable target signal in the
  trajectory alone.
- **Input cleaning helps, trimming is the active ingredient.** Trim + per-session
  rescale raises the open-loop gain +1.10 → +1.43 pp and halves across-seed
  variance (`studies/experiment_clean.py`, `studies/ablate_cleaning.py`). Cleaning
  is metric-safe: the raw-decoder accuracy is identical to the decimal before and
  after (verifiable in one line — see Quickstart).
- **Feature engineering is closed.** No richer feature set beats the current one by
  more than seed noise; the signal lives in per-tick *velocity*, not accumulated
  position (`studies/feature_sweep.py`).

### Open items (things the code does not yet reflect)

1. **Cleaning is opt-in only.** `cd.load_source(..., clean=True)` exists and is used
   by the studies, but `train_copilot.py` / `evaluate_copilot.py` / `closed_loop.py`
   have **no `--clean` flag** — the production path still trains on raw inputs. The
   7/29 recommendation is to make cleaned inputs the default input path.
2. **Baselines pending re-run on the fixed loader.** The training-data composition
   (blend) study, the correctability-ceiling diagnostic, and the 7/22 probes were
   all measured on the corrupted trial set. Their numbers should not be quoted
   until re-run. See `PIPELINE.md` §7.

## Repository map

### Root — the pipeline (this is what you read)
| File | Role |
| --- | --- |
| `copilot_dataset.py` | Data foundation: loading, cleaning, the canonical accuracy/angle metric, feature building, normalization, and the leakage-free `split_real`. |
| `copilot_core.py` | The model (`LSTMCopilot`) and the Stage-2 corrective-velocity control law. Single source of truth; imported everywhere. |
| `train_copilot.py` | Trains a copilot (Phase-1 teacher-forced features + Phase-2 DAgger on-policy rollouts). |
| `evaluate_copilot.py` | Authoritative **open-loop** held-out evaluation of a trained run. |
| `sim_scaling.py` | Calibrates EEGK-sim onto real (radius scaling + onset dwell). |
| `blend_constructor.py` | Builds fixed-budget real/sim training blends. |
| `closed_loop.py` | Per-subject closed-loop surrogate environment + the 3-leg validation harness (`PIPELINE.md` §5). |
| `trajectory_aware_copilot.py` | Causal accumulated-belief steering + open-loop A/B (fails open-loop; predicted to pay off closed-loop). |

### `studies/` — closed investigations
Self-contained A/Bs and diagnostics answering one question each, each reusing the
production modules. `diagnose_learning.py` (is the model sound?), `feature_sweep.py`
(which features carry the signal?), `experiment_clean.py` (does cleaning move the
gain?), `ablate_cleaning.py` (trim vs rescale). See `studies/README.md`.

### `analysis/` — characterization tools
`profile_sources.py` (per-subject spatial/temporal/kinematic trajectory stats) and
`visualize_sources.py` (HTML trajectory viewer). Neither is on the training path.

### `experimental/` — Problem 2 seeds and probes
Language-model word/sentence decoding (`word_decode.py`, `sentence_decode.py`,
`sentence_rerank.py`, `prior_fusion_probe.py`), the reactive closed-loop variant
(`closed_loop_reactive.py`), and the readout probe. Not on the critical path;
`fusion_pipeline.py` is not runnable as-is. See `experimental/README.md`.

### `archive/` — retired but runnable
Endpoint-modeled surrogate, blend-ratio sweep, residual/ceiling diagnostic. Kept
for provenance; nothing active imports them. See `archive/README.md`.

### Legacy (reference only)
- `legacyRL/` — the Lee et al. RL-copilot infrastructure (PPO, `env.py`, data-driven
  surrogate). Abandoned for the supervised approach; see `legacyRL/README_legacy.md`.
  Its closed-loop `env.py` is a useful reference for `closed_loop.py`.
- `legacySL/` — earlier supervised-copilot scripts and the V3 model.

### `data/`
`OnlineArmTrajectoryEEGK/` — real EEGK online trajectories: 6 subjects
(S01–S05, S07), **22 session folders, 16,197 trials**.
`online_arm_trajectories_EEGK_simulation.csv` — re-decoded sim.
`online_arm_trajectories.csv` — the older decoder's trajectories.
Schema: `data/README_online_arm_trajectories.md`.

Generated artifacts (`runs/`, `results/`, `data/blend/`, `data/surrogate/`) are
git-ignored and regenerable.

## Quickstart

Python 3.12 with numpy, pandas, torch (see `requirements.txt`).

```bash
# 1. sanity-check the data foundation (16,197 trials, 64.04% raw baseline)
python -c "import copilot_dataset as cd; r=cd.load_source('eegk_real'); print(len(r), cd.baseline_metrics(r))"
```

```bash
# 2. confirm cleaning is metric-safe (identical accuracy, raw vs cleaned)
python -c "import copilot_dataset as cd; print(cd.baseline_metrics(cd.load_source('eegk_real')), cd.baseline_metrics(cd.load_source('eegk_real', clean=True)))"
```

```bash
# 3. train one copilot per subject on real data, then evaluate open-loop on held-out blocks
python train_copilot.py --training_data eegk_real --model_type sl --train_test within_subject --copilot_vel_mag 0.02
```

```bash
python evaluate_copilot.py --run runs/<run_dir> --eval_data eegk_real --eval_split test --split_seed 0
```

```bash
# 4. the closed-loop surrogate + its 3-leg validation
python closed_loop.py --seed 0 --n_trials 500
```

```bash
# 5. reproduce the 7/29 diagnostics
python studies/diagnose_learning.py
```

```bash
python studies/experiment_clean.py --seeds 0 1 2
```

## Conventions

- **Flat import namespace.** Everything imports root modules by bare name
  (`import copilot_dataset as cd`). Scripts living in `studies/`, `analysis/`,
  `experimental/`, and `archive/` carry a two-line `sys.path` shim at the top so
  they run unchanged from the repo root: `python studies/feature_sweep.py`.
- **Every script is an entry point** with a module docstring stating the question it
  answers and a `RUN` section. Reading the docstring before the code is the
  intended path.
- **The metric is direction-only** (argmax dot-product on the final cursor
  position), so transforms that preserve direction cannot change a label — that is
  what makes the cleaning A/B valid.
