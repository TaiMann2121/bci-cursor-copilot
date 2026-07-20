# experimental/

Seeds for **Problem 2** (language-model-assisted word/sentence decoding) and other
ideas not yet integrated into the copilot pipeline. Nothing here is on the critical
path; treat it as a scratchpad.

## `fusion_pipeline.py`
A first pass at fusing **motor evidence** (per-character arm/finger posteriors from
real data) with a **character-level 4-gram language model** to decode groups,
characters, and whole words. It demonstrates the "motor × language prior" idea from
the project brief (Problem 2).

**Not runnable as-is:**
- Imports `from harness import load_aligned, TA, nd` — `harness.py` was never
  committed to this repo. The needed pieces (aligned real trajectories + target
  angles) are available via `copilot_dataset.load_source("eegk_real")`; porting
  `fusion_pipeline.py` onto `copilot_dataset` is the first step to reviving it.
- Requires `pip install wordfreq`.

**Why it's kept:** the eventual copilot should condition its target inference on a
language-model prior over the next direction given previously typed keys (the
Problem 2 → Problem 1 bridge described in `PIPELINE.md` §4). This file is the seed
for that prior.
