# analysis/

Characterization tools. Neither script trains or evaluates a copilot; both describe
what the trajectory data *is*, which is how the sim-vs-real calibration decisions
were made. Run from the repo root:

```bash
python analysis/profile_sources.py
```

| File | What it produces |
| --- | --- |
| `profile_sources.py` | Per-subject, per-direction trajectory statistics across three families — spatial (endpoint radius, endpoint angle error), temporal (length, dwell ticks), kinematic (step magnitude mean/std, wander index, reversal rate) — with real as the reference and each synthetic source beside it. Flags which properties are decoder-shared (real ≈ sim) vs subject-specific (real ≠ sim). Optional `.xlsx` output if `openpyxl` is installed. |
| `visualize_sources.py` | An HTML grid of overlaid trajectories (subject × direction, real vs sim), for eyeballing what the statistics summarize. Writes to `results/` (git-ignored). |

Both read through `copilot_dataset.load_source` and apply the same `sim_scaling`
calibration the training path uses, so what you see is what the model sees.
