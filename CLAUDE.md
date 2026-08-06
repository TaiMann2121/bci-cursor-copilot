# CLAUDE.md

An AI **copilot** for the arm phase of an EEG-BCI typing system. EEGNet decodes
noisy per-tick cursor velocities, the final cursor position selects one of 8
directions, and the copilot observes the decoded trajectory, infers the intended
target, and adds a corrective velocity.

## Orientation

Read in this order. Do not re-derive project state from code or git history —
these files are kept current on purpose.

| File | What it tells you |
| --- | --- |
| `ROADMAP.md` | **What to work on next.** Current priorities, task status, blocking dependencies. Supersedes `PIPELINE.md` §8. |
| `PIPELINE.md` | *Why* the architecture looks like this, and which conclusions currently stand (§7 table). |
| `README.md` | Status summary + repository map. |
| `READING_GUIDE.md` | Staged code walkthrough with file:line anchors. |

## Commands

```bash
# sanity-check the data foundation -> expect 16197 trials, 0.6404 accuracy
python -c "import copilot_dataset as cd; r=cd.load_source('eegk_real'); print(len(r), cd.baseline_metrics(r))"

# train one copilot per subject, then evaluate open-loop on held-out blocks
python train_copilot.py --training_data eegk_real --model_type sl --train_test within_subject --copilot_vel_mag 0.02
python evaluate_copilot.py --run runs/<run_dir> --eval_data eegk_real --eval_split test --split_seed 0

# is a gain real? (the null control -- see house rules below)
python studies/null_control.py --seed 0 --clean --sim_frac 0
python studies/aggregate_null_control.py "results/null_control_randsplit_seed*.log"
```

Scripts in `studies/`, `analysis/`, `experimental/`, and `archive/` run **from the
repo root**: `python studies/feature_sweep.py`.

## Conventions

- **Flat import namespace.** Everything imports root modules by bare name
  (`import copilot_dataset as cd`). Subdirectory scripts carry a two-line
  `sys.path` shim so they run unchanged from the repo root.
- **Every script is an entry point** whose module docstring states the question it
  answers and how to run it. Read the docstring before the code.
- **The metric is direction-only** (argmax dot-product on the final cursor
  position). Transforms that preserve direction cannot change a label — that is
  what makes the cleaning A/B valid.
- **`copilot_core.py` is the single source of truth** for the model and control
  law. Do not reimplement either in a study; import them.

## House rules for measurement

These come from 8/4, when a day of single-run readings pointed the wrong way four
times before an 8-split null control settled the question.

1. **Report gains against a null and across splits.** One split read +0.85 pp
   against a +0.28 pp null and looked like noise; 8 splits put the null at ~0 and
   the effect at +1.01 ± 0.39 pp. `studies/null_control.py` is the template.
2. **Never compare a random-split number to a session-blocked one.**
   `studies/diagnose_learning.py` uses a *stratified random* split, so trials from
   one session land in both train and val — it measures **fitting**, not
   cross-session generalization. `cd.split_real` holds out whole (session, run)
   blocks. Conflating the two started the 8/4 investigation.
3. **There is no single baseline number.** Pooled over all 16,197 trials it is
   **64.04%**, but a *held-out split* baseline ranges **63.3–66.6%**. Single-split
   headroom arguments carry ±1.4 pp.
4. **Check for mode collapse before trusting a classifier readout.** If final-tick
   and belief-aggregated accuracy are identical and match the most frequent test
   label's frequency, the model collapsed to a constant. This silently produced 3
   of 4 broken configs before the 8/4 fix.

## Gotchas

- **`train_one_model` selects on `val_ce`** (`--select_on`). It used to select on
  validation *copilot endpoint* accuracy while training on per-tick CE; those are
  decoupled, so mode-collapsed classifiers were saved as "best". Do not revert to
  `copilot` without reading `PIPELINE.md` §2.6.
- **`split_real` randomizes test blocks** (`random_test_blocks=True`, the default).
  Pass `False` only to reproduce pre-8/4 numbers — everything measured before 8/4
  used one deterministic held-out set.
- **Cleaning is opt-in.** `cd.load_source(..., clean=True)` exists, but
  `train_copilot.py` / `evaluate_copilot.py` / `closed_loop.py` have **no
  `--clean` flag**; the production path still trains on raw inputs.
- **`closed_loop.train_copilots` bypasses `train_copilot.main()`**, so anything
  `main()` sets up (torch seeding, `--fixed_norm`) must be handled there
  explicitly.
- **Windows console is cp1252.** Non-ASCII in printed output (`Δ`, `±`, `→`)
  crashes a script at the print, after the compute has already run. Keep script
  *output* ASCII; markdown files are fine.
- **Results are regenerable and git-ignored** (`results/`, `runs/`, `data/blend/`,
  `data/surrogate/`). `.gitignore` has no inline-comment syntax — put comments on
  their own line or the pattern silently matches nothing.

## Data

`data/OnlineArmTrajectoryEEGK/` — 6 subjects (S01–S05, S07), 22 session folders,
**16,197 trials**, committed to the repo (~42 MB). Trial keys are *not* unique
across session folders, which is why trajectories carry `session_id` and real data
is always loaded per folder; see `PIPELINE.md` §2.5.
