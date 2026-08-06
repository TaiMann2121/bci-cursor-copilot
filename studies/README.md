# studies/

Closed investigations. Each script answers **one question**, reuses the production
modules (`copilot_dataset`, `copilot_core`, `train_copilot`) rather than
reimplementing them, and states its question in its module docstring. They are run
from the repo root — each carries a two-line `sys.path` shim so bare imports resolve:

```bash
python studies/diagnose_learning.py
```

Read them in this order; each depends on the one above it.

| File | Question | Answer |
| --- | --- | --- |
| `null_control.py` | **Is the copilot's gain real?** Drives the *same* control law with classifiers carrying no target information — random-init, shuffled-label, and a constant predictor — on the same held-out trials. | **Yes (8/4).** Paired trained − shuffled = **+1.01 ± 0.39 pp, positive on 8/8 randomized splits** (t = 7.2). A confident but permanently wrong pusher loses 6.75 pp, so the control law is not free lunch. Pool per-split logs with `aggregate_null_control.py`. |
| `aggregate_null_control.py` | Companion to the above: pools per-split logs into the **paired** trained-vs-null statistic. Pairing is within-split, so both arms share the same test trials and baseline. | Reports mean ± sd per condition, the paired difference, and an indicative CI. |
| `diagnose_learning.py` | Is the LSTM functionally sound, and is there learnable target signal in the trajectory at all? Watches train/val cross-entropy, not downstream accuracy. | Yes. Train CE falls far below chance (ln 8 = 2.079) for all six subjects, validation tracks it, final-tick target accuracy 57–78% vs 12.5% chance. **⚠️ Qualified 8/4: this uses a *stratified random* split, so trials from one session appear in both train and val. It measures *fitting*, not cross-session generalization — do not compare its numbers against session-blocked ones (`split_real`).** |
| `feature_sweep.py` | *Which* trajectory information carries that signal? Six feature-group configs, plain 8-way classifier, no control law. | Position-only is ~6σ worse than anything with velocity; velocity-only matches the full set. Signal is in per-tick velocity. No richer set beats `basic` by more than seed noise — the item is closed. |
| `experiment_clean.py` | Does feeding cleaned inputs move the copilot's **open-loop gain**? Full pipeline, identical control law, only the inputs differ. | +1.10 → +1.43 pp, and across-seed variance halves (±0.30 → ±0.16). |
| `ablate_cleaning.py` | How does that split between trimming and per-session rescaling, and does the scale *reference* matter? | Trimming is the active ingredient (+0.17 pp alone); rescaling alone is −0.24 pp; together +0.33 pp — they interact. Global vs per-subject reference differs by 0.07 pp (inside noise); per-subject is the default since the pipeline trains one model per subject. |

`feature_sweep.py` imports `diagnose_learning`, and `ablate_cleaning.py` imports
`experiment_clean` (same folder, so bare imports work) — deliberately, so the
protocol being compared is literally the same code.

## Caveat on the 7/29 rows (read before quoting them)

`feature_sweep.py`, `experiment_clean.py`, and `ablate_cleaning.py` all predate
the **8/4 checkpoint-selection fix**, under which `train_one_model` could save a
mode-collapsed classifier as "best" (see `PIPELINE.md` §2.6). They also predate
`split_real(random_test_blocks=True)`, so each was measured on a single
deterministic held-out set. Re-run before quoting; `ROADMAP.md` tracks which
re-runs matter and which are deliberately skipped.

## House rule for any new study here

Report gains **against a null and across splits**. On 8/4 a single split pointed
the wrong way twice in one day: it read +0.85 pp against a +0.28 pp null and
looked like noise, while 8 splits put the null at ~0 and the effect at
+1.01 ± 0.39 pp. `null_control.py` is the template.
