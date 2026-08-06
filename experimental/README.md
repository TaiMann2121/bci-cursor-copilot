# experimental/

Probes and seeds that are **not on the critical path**: Problem 2 (language-model
word/sentence decoding) plus a few one-off measurements of Problem 1. Nothing here
is imported by the production pipeline. Each script carries a `sys.path` shim and
runs from the repo root:

```bash
python experimental/readout_probe.py --seed 0
```

## Problem 1 probes (arm phase)

| File | Question |
| --- | --- |
| `readout_probe.py` | Is the *control law* the bottleneck, or the classifier? Compares raw-BCI endpoint argmax, the copilot rollout, and two direct classifier readouts (final-tick argmax, belief-aggregated argmax) on the same held-out trials, then asks which is right on the trials where they disagree. **Answer (8/4, post-fix): neither — the classifier readout sits at *parity* with the raw endpoint (65.1% vs 65.1%), disagreements 37% vs 42%.** Flags: `--clean`, `--sim_frac` (0 = real-only), `--eval_norm {test,train}`. Use `--eval_norm train` to avoid the train/eval norm mismatch documented at `train_copilot.py --fixed_norm`; the `test` default preserves the house convention so older logs reproduce. |
| `closed_loop_reactive.py` | The credible closed-loop test. Replaces the parametric surrogate in `../closed_loop.py` with a data-driven one (real per-tick control errors replayed from the *current* cursor, so leg-1 faithfulness is automatic) and adds **user persistence α**: α=0 is open-loop replay, α=1 is fully closed-loop. Sweeping α maps how much of the copilot's benefit depends on the user reacting — that dependence is the result. |
| `prior_fusion_probe.py` | Does a language prior over the next *direction* beat motor-alone on the arm phase? Fuses a von-Mises motor posterior around the raw endpoint angle with an English group n-gram, under oracle and decoded context. |

## Problem 2: language-assisted decoding

| File | Question |
| --- | --- |
| `word_decode.py` | Can a language model recover intended **words** from noisy per-character motor evidence? A character is (arm group, finger); per-character accuracy is mediocre, so motor-only whole-word accuracy is tiny — but spelling + word-frequency constraints recover far above the per-character rate. Uses real arm endpoints and the real finger confusion matrix from `typing_stats.npz`. |
| `sentence_decode.py` | The same at the **sentence** level, where segmentation is part of the problem (space shares group 5 with b/n). A beam-search noisy-channel decoder over the full character sequence, scoring motor + char-LM + word-validity jointly. Compared against motor-only and word-level decoding. |
| `sentence_rerank.py` | The next lever above the DP decoder: give an LLM the known character length, candidate decodes, and top-k per-position symbols, and ask for the most likely English sentence. Needs `ANTHROPIC_API_KEY` or an `ant auth login` profile; `--mock` exercises the plumbing with no API calls. Responses are cached in `.rerank_cache.json` (git-ignored). |
| `fusion_pipeline.py` | The original motor × character-4-gram fusion sketch. **Not runnable as-is:** it imports `from harness import load_aligned, TA, nd` and `harness.py` was never committed. The pieces it needs (aligned real trajectories + target angles) are available via `copilot_dataset.load_source("eegk_real")`; porting it onto `copilot_dataset` is step one to reviving it. Also requires `pip install wordfreq`. |

## Why these are kept

The eventual copilot should condition its target inference on a language-model prior
over the next direction given previously typed keys — the Problem 2 → Problem 1
bridge in `PIPELINE.md` §4. `prior_fusion_probe.py` measures what that is worth at
the direction level; `word_decode.py` / `sentence_decode.py` show where the larger
headroom is.

**As of 8/3 this folder is no longer off the critical path.** The supervisor
green-lit the pivot to a joint language + trajectory copilot, so these probes are
prototypes of the new architecture rather than side experiments. `ROADMAP.md`
Phase 2 identifies the biggest unexploited lever: the user is constrained to a
**closed word library with known word length**, so the prior over the next
direction should be a **trie over library words still consistent with the prefix
typed so far** — far sharper than the generic English character n-gram these
probes use, and usable from tick 0 because it is fixed for a character's whole
arm phase. Re-running `prior_fusion_probe.py` is ROADMAP task 0.2, the next item.

**Caveat:** every number produced in this folder except `readout_probe.py`'s
predates the 7/29 loader fix *and* the 8/4 checkpoint-selection fix, so it may
have been measured on both a corrupted trial set and a mode-collapsed model
(`PIPELINE.md` §2.5, §2.6). Pending re-run.
