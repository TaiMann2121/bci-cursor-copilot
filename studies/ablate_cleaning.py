"""
ablate_cleaning.py
==================
Decomposes the cleaning result from experiment_clean.py into its parts, and
tests a confound found on review.

Two questions:

  Q1  How does the +0.40 pp cleaning gain split between the two steps the
      supervisor named separately -- trimming the leading dead ticks, and
      per-session scale normalization? experiment_clean.py applied both at once.

  Q2  Does the SCALE REFERENCE matter? Global cleaning equals per-subject
      cleaning times a per-subject constant c_S = T_global / T_subject. That
      constant does NOT cancel: the copilot's corrective magnitude
      (copilot_vel_mag = 0.02) is ABSOLUTE while the BCI velocity scales with
      c_S, so a global reference silently changes the copilot's effective
      strength per subject (~35% spread across subjects, weakest for S07 and
      S01 -- which are exactly the two subjects where cleaning appeared to
      hurt). Since the pipeline trains one model per subject, 'per_subject' is
      the principled reference; this measures how much the choice mattered.

Variants (all leakage-free, real-only, identical control law, multi-seed):
    raw               no cleaning
    trim_only         trim leading dwell, no rescaling
    scale_global      rescale with the GLOBAL reference, no trimming
    scale_persubj     rescale with the PER-SUBJECT reference, no trimming
    both_global       trim + global rescale      (what experiment_clean ran)
    both_persubj      trim + per-subject rescale (the corrected default)

RUN
---
    python studies/ablate_cleaning.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# repo root on sys.path so the pipeline modules (copilot_dataset, ...) import
# unchanged when this script is run from here: python studies/ablate_cleaning.py
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.dirname(_HERE), _HERE]   # repo root, then this folder

import copilot_dataset as cd
from experiment_clean import CFG, train_and_eval_variant  # reuse the exact protocol

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# variant -> (clean_flag, clean_kwargs)
VARIANTS = {
    "raw":           (False, None),
    "trim_only":     (True,  dict(do_trim=True,  do_scale=False)),
    "scale_global":  (True,  dict(do_trim=False, do_scale=True, reference="global")),
    "scale_persubj": (True,  dict(do_trim=False, do_scale=True, reference="per_subject")),
    "both_global":   (True,  dict(do_trim=True,  do_scale=True, reference="global")),
    "both_persubj":  (True,  dict(do_trim=True,  do_scale=True, reference="per_subject")),
}


def run(seeds, subjects):
    res = {}
    for name, (clean, kw) in VARIANTS.items():
        res[name] = []
        for seed in seeds:
            print(f"  {name:>14s} seed={seed} ...", flush=True)
            res[name].append(train_and_eval_variant(clean, seed, subjects, clean_kwargs=kw))

    print("\n" + "=" * 84)
    print(f"CLEANING ABLATION (open-loop, leakage-free, real-only)  seeds={seeds}")
    print(f"  control law identical: vel_mag={CFG['copilot_vel_mag']}, "
          f"features={CFG['input_feature_set']}, mode={CFG['velocity_mode']}")
    print("=" * 84)
    print(f"{'variant':>14} {'BCI acc':>9} {'copilot':>9} {'gain':>16} {'vs raw':>9}")
    print("-" * 84)
    base = None
    for name in VARIANTS:
        runs = res[name]
        bci = np.array([r["overall"]["bci"] for r in runs])
        cop = np.array([r["overall"]["cop"] for r in runs])
        g = (cop - bci) * 100
        if base is None:
            base = g.mean()
        print(f"{name:>14} {bci.mean()*100:>8.2f}% {cop.mean()*100:>8.2f}% "
              f"{g.mean():>+11.2f}±{g.std():.2f}pp {g.mean()-base:>+8.2f}")
    print("-" * 84)

    print("\nper-subject gain (pp, mean over seeds):")
    subs = sorted(res["raw"][0]["per_subject"])
    print(f"{'subj':>5} " + " ".join(f"{n:>14}" for n in VARIANTS))
    for s in subs:
        row = []
        for name in VARIANTS:
            g = np.mean([(r["per_subject"][s]["cop"] - r["per_subject"][s]["bci"]) * 100
                         for r in res[name]])
            row.append(f"{g:>+14.2f}")
        print(f"{s:>5} " + " ".join(row))
    print("=" * 84)
    print("Q1: trim_only vs scale_* vs both_*  -> how the gain splits between the two steps")
    print("Q2: both_global vs both_persubj     -> whether the reference choice confounded")
    print("    the reported +0.40 pp (watch S01 and S07, the weakened-copilot subjects)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()
    run(args.seeds, args.subjects)


if __name__ == "__main__":
    main()
