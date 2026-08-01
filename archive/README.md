# archive/

Root scripts retired from the active pipeline. They still run (each has a two-line
`sys.path` shim so its `import copilot_dataset` etc. resolve when executed from
here — e.g. `python archive/diagnose_residuals.py`), but they are off the critical
path and kept for provenance. Nothing in the active pipeline imports them.

| File | Why retired |
| --- | --- |
| `surrogate_constructor.py` | The **endpoint-modeled surrogate** — the *surrogate-as-training-data* approach. Deprioritized in the 7/13 report (it reproduces endpoint statistics but not the target-dependent structure the copilot needs) and superseded by the *surrogate-as-closed-loop-environment* in `../closed_loop.py`. |
| `sweep_blends.py` | The real:sim blend-ratio sweep. Its finding (calibrated sim helps only as a ~25% minority augmenter) is banked in the progress reports; the training-data composition question is settled. |
| `diagnose_residuals.py` | The correctability-ceiling / residual-error diagnostic. Its finding (~87% of decoder errors carry no recoverable intent open-loop) is banked and drives the closed-loop decision; kept as the reproducible source behind that claim. |

To resurrect one, move it back to the repo root and delete its `sys.path` shim.

## Note (7/29/2026): two of these are the blocking re-runs

The findings banked from `sweep_blends.py` (sim helps only as a ~25% minority
augmenter) and `diagnose_residuals.py` (~87% of decoder errors carry no recoverable
intent) were both measured **before the loader defect was found** — on a trial set in
which 3,030 pairs of different-target trials had been fused. Both are listed as
*pending re-run* in `PIPELINE.md` §7, and re-running them on the fixed loader is the
blocking item in §8. So these two scripts are retired from the *pipeline*, but not
retired as *evidence*: they are the reproducible source for claims that currently
need re-establishing.
