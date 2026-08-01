# studies/

Closed investigations. Each script answers **one question**, reuses the production
modules (`copilot_dataset`, `copilot_core`, `train_copilot`) rather than
reimplementing them, and states its question in its module docstring. They are run
from the repo root — each carries a two-line `sys.path` shim so bare imports resolve:

```bash
python studies/diagnose_learning.py
```

Read them in this order; each depends on the one above it.

| File | Question | Answer (7/29/2026) |
| --- | --- | --- |
| `diagnose_learning.py` | Is the LSTM functionally sound, and is there learnable target signal in the trajectory at all? Watches train/val cross-entropy, not downstream accuracy. | Yes. Train CE falls far below chance (ln 8 = 2.079) for all six subjects, validation tracks it, final-tick target accuracy 57–78% vs 12.5% chance. |
| `feature_sweep.py` | *Which* trajectory information carries that signal? Six feature-group configs, plain 8-way classifier, no control law. | Position-only is ~6σ worse than anything with velocity; velocity-only matches the full set. Signal is in per-tick velocity. No richer set beats `basic` by more than seed noise — the item is closed. |
| `experiment_clean.py` | Does feeding cleaned inputs move the copilot's **open-loop gain**? Full pipeline, identical control law, only the inputs differ. | +1.10 → +1.43 pp, and across-seed variance halves (±0.30 → ±0.16). |
| `ablate_cleaning.py` | How does that split between trimming and per-session rescaling, and does the scale *reference* matter? | Trimming is the active ingredient (+0.17 pp alone); rescaling alone is −0.24 pp; together +0.33 pp — they interact. Global vs per-subject reference differs by 0.07 pp (inside noise); per-subject is the default since the pipeline trains one model per subject. |

`feature_sweep.py` imports `diagnose_learning`, and `ablate_cleaning.py` imports
`experiment_clean` (same folder, so bare imports work) — deliberately, so the
protocol being compared is literally the same code.
