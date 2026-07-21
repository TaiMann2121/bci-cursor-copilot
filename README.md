# arm-bci-copilot

An AI **copilot** for the arm phase of an EEG-BCI typing system. In the two-stage
paradigm, EEGNet decodes noisy per-tick cursor velocities and the final cursor
position selects one of 8 directions; the copilot observes the decoded trajectory,
infers the intended target, and adds a corrective velocity to aid the cursor.

> **Read [`PIPELINE.md`](PIPELINE.md) first.** It states the current architecture
> decision — moving from open-loop replay evaluation to a **closed-loop
> subject-surrogate** — and the argument for why gains should transfer to human
> trials. Everything below is the map.

## Status (current)

- **Control law is settled:** supervised LSTM target classifier → additive
  corrective velocity (`copilot_core.py`).
- **Open-loop gain is small and has plateaued:** ~**+1 pp** endpoint accuracy over
  raw decoder (raw 53.3% → copilot 54.3%, current 6-subject data, seed 0). The
  training-data composition study is exhausted (calibrated sim helps only as a
  ~25% minority augmenter). See `Progress Reports/`.
- **Active direction:** closed-loop evaluation against a calibrated per-subject
  surrogate (`closed_loop.py`), because open-loop replay structurally cannot
  reward a copilot's feedback benefit. See `PIPELINE.md` §3–5.

## Repository map

### Core pipeline (the product — start here)
| File | Role |
| --- | --- |
| `copilot_core.py` | The model (`LSTMCopilot`) and the Stage-2 corrective-velocity controller. Single source of truth for the control law; imported everywhere. |
| `copilot_dataset.py` | Data foundation: trajectory parsing, the canonical accuracy/angle metric, normalization, and the leakage-free `split_real`. |
| `train_copilot.py` | Trains a copilot (Phase-1 BCI features + Phase-2 DAgger augmentation). |
| `evaluate_copilot.py` | Authoritative **open-loop** held-out evaluation of a trained run. |
| `blend_constructor.py` | Builds fixed-budget real/sim training blends. |
| `sim_scaling.py` | Calibrates EEGK-sim onto real (radius scaling + onset dwell). |
| **`closed_loop.py`** | **The new direction:** per-subject closed-loop surrogate environment + the 3-leg validation harness (`PIPELINE.md` §5). |
| `trajectory_aware_copilot.py` | Causal accumulated-belief steering + open-loop A/B (fails open-loop; predicted to pay off closed-loop). |

### Analysis & diagnostics
| File | Role |
| --- | --- |
| `profile_sources.py` | Per-subject trajectory characterization (spatial/temporal/shape). |
| `visualize_sources.py` | HTML trajectory visualizations. |

### Archive (`archive/`)
Retired-but-runnable scripts, off the critical path (nothing active imports them):
the endpoint-modeled surrogate (`surrogate_constructor.py`, superseded by
`closed_loop.py`), the blend-ratio sweep (`sweep_blends.py`), and the ceiling
diagnostic (`diagnose_residuals.py`). See `archive/README.md`.

### Experimental (`experimental/`)
Problem-2 (language-model) seeds, not yet wired into the copilot. `fusion_pipeline.py`
fuses motor evidence with a character LM; **it is not runnable as-is** (needs an
un-committed `harness.py` and `pip install wordfreq`) — see `experimental/README.md`.

### Legacy (reference only; nothing active imports these)
- `legacyRL/` — the Lee et al. RL-copilot infrastructure (PPO, `env.py`, the
  data-driven surrogate). Abandoned for the supervised approach; see
  `legacyRL/README_legacy.md`. Its closed-loop `env.py` is a useful reference for
  `closed_loop.py`.
- `legacySL/` — earlier supervised-copilot scripts and the V3 model.

### Data (`data/`)
`OnlineArmTrajectoryEEGK/` — real EEGK online trajectories (6 subjects: S01–S05,
S07) + per-session `typing_stats.npz`. `online_arm_trajectories_EEGK_simulation.csv`
— re-decoded sim. See `data/README_online_arm_trajectories.md` for the schema.
Generated artifacts (`runs/`, `results/`, `data/blend/`, `data/surrogate/`) are
git-ignored and regenerable.

## Quickstart

```bash
# environment: python3.12 with torch, pandas, numpy, scikit-learn (see requirements.txt)
python closed_loop.py --seed 0 --n_trials 500     # closed-loop surrogate + validation legs
python trajectory_aware_copilot.py --seed 0        # open-loop trajectory-aware A/B
python evaluate_copilot.py --run runs/<run_dir> --eval_data eegk_real   # open-loop eval
```
