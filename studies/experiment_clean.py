"""
experiment_clean.py
===================
Does feeding CLEANED inputs move the copilot's open-loop gain?

Leakage-free, real-only, within-subject A/B:
  for variant in {raw, clean}:
      real   = load_source('eegk_real', clean=variant)   # same 16,197 trials
      splits = split_real(seed)                           # whole-block held-out test
      per subject: train copilot (DAgger) on TRAIN, eval OPEN-LOOP on TEST
      report raw-BCI acc, copilot acc, Δ(copilot - BCI)

The copilot control law (additive corrective velocity, vel_mag, features, epochs)
is held IDENTICAL across variants; only the input trajectories differ (raw vs
trim+per-session-scale). Because cleaning is metric-safe, the raw-BCI accuracy is
identical across variants by construction — so any change in Δ is attributable to
the copilot reading cleaner features/dynamics.

Multi-seed; reports mean±sd of Δ per variant.

RUN
---
    python studies/experiment_clean.py --seeds 0 1 2
    python studies/experiment_clean.py --seeds 0 --subjects S04 S05
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# repo root on sys.path so the pipeline modules (copilot_dataset, ...) import
# unchanged when this script is run from here: python studies/experiment_clean.py
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.dirname(_HERE), _HERE]   # repo root, then this folder

import copilot_core as core
import copilot_dataset as cd
import train_copilot as tc

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CFG = dict(copilot_vel_mag="0.02", input_feature_set="basic", velocity_mode="additive",
           tick_weighting="exponential", weight_exponent=3.0, num_lstm=2, hidden_size=64,
           learning_rate=1e-3, grad_clip=1.0, batch_size=128, val_frac=0.15)


def train_and_eval_variant(clean: bool, seed: int, subjects, clean_kwargs=None):
    """Train per-subject copilots on TRAIN, eval open-loop on TEST. Returns
    {'overall': {...}, 'per_subject': {s: {...}}}.

    clean_kwargs is forwarded to the cleaner (do_trim / do_scale / reference),
    which is how ablate_cleaning.py decomposes the cleaning effect."""
    real = cd.load_source("eegk_real", clean=clean, clean_kwargs=clean_kwargs)
    splits = cd.split_real(real, seed=seed)

    agg = {"n": 0, "bci": 0, "cop": 0}
    per = {}
    for subj in sorted(splits):
        if subjects and subj not in subjects:
            continue
        train = splits[subj]["train"]
        test = splits[subj]["test"]
        if not train or not test:
            continue
        norm_tr = cd.compute_norm_stats(train)
        views = [tc.trial_view(t) for t in train]

        torch.manual_seed(seed); np.random.seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            d = Path(tempfile.mkdtemp())
            tc.train_one_model(views, norm_tr, CFG, subj, d)
            model = core.LSTMCopilot(input_size=5, hidden_size=64, n_layers=2)
            model.load_state_dict(torch.load(d / "best_model.pt")); model.eval()

        norm_te = cd.compute_norm_stats(test)
        vels = [(t.pos[1:] - t.pos[:-1]).astype(np.float32) for t in test]
        labels = np.array([t.target_label for t in test])
        bci = np.array([cd.label_from_position(t.final_pos) for t in test])
        _, _, cop = core.simulate_batch(model, vels, norm_te, CFG["copilot_vel_mag"],
                                        CFG["input_feature_set"], "cpu", CFG["velocity_mode"])
        n = len(test)
        per[subj] = {"n": n,
                     "bci": float((bci == labels).mean()),
                     "cop": float((np.array(cop) == labels).mean())}
        agg["n"] += n
        agg["bci"] += int((bci == labels).sum())
        agg["cop"] += int((np.array(cop) == labels).sum())
    overall = {"n": agg["n"], "bci": agg["bci"] / agg["n"], "cop": agg["cop"] / agg["n"]}
    return {"overall": overall, "per_subject": per}


def run(seeds, subjects):
    res = {"raw": [], "clean": []}
    for seed in seeds:
        for variant, clean in (("raw", False), ("clean", True)):
            print(f"  running variant={variant} seed={seed} ...", flush=True)
            res[variant].append(train_and_eval_variant(clean, seed, subjects))

    print("\n" + "=" * 78)
    print(f"CLEAN vs RAW copilot (open-loop, leakage-free, real-only)  seeds={seeds}")
    print(f"  control law identical (vel_mag={CFG['copilot_vel_mag']}, "
          f"features={CFG['input_feature_set']}, mode={CFG['velocity_mode']})")
    print("=" * 78)
    print(f"{'variant':>8} {'BCI acc':>9} {'copilot':>9} {'Δ (cop-BCI)':>13}")
    print("-" * 78)
    summary = {}
    for variant in ("raw", "clean"):
        runs = res[variant]
        bci = np.array([r["overall"]["bci"] for r in runs])
        cop = np.array([r["overall"]["cop"] for r in runs])
        dl = (cop - bci) * 100
        summary[variant] = (bci.mean(), cop.mean(), dl.mean(), dl.std())
        print(f"{variant:>8} {bci.mean()*100:>8.2f}% {cop.mean()*100:>8.2f}% "
              f"{dl.mean():>+8.2f}±{dl.std():.2f}pp")
    print("-" * 78)
    d_raw, d_clean = summary["raw"][2], summary["clean"][2]
    print(f"copilot gain:   raw Δ={d_raw:+.2f}pp   clean Δ={d_clean:+.2f}pp   "
          f"(clean - raw = {d_clean - d_raw:+.2f}pp)")
    print(f"absolute copilot acc: raw {summary['raw'][1]*100:.2f}%  ->  "
          f"clean {summary['clean'][1]*100:.2f}%  ({(summary['clean'][1]-summary['raw'][1])*100:+.2f}pp)")

    # per-subject (mean over seeds), clean variant
    print("\nper-subject (mean over seeds):")
    print(f"{'subj':>5} {'raw BCI':>8} {'raw cop':>8} {'rawΔ':>7} | "
          f"{'cln BCI':>8} {'cln cop':>8} {'clnΔ':>7}")
    subs = sorted(res['clean'][0]['per_subject'])
    for s in subs:
        rb = np.mean([r['per_subject'][s]['bci'] for r in res['raw']]) * 100
        rc = np.mean([r['per_subject'][s]['cop'] for r in res['raw']]) * 100
        cb = np.mean([r['per_subject'][s]['bci'] for r in res['clean']]) * 100
        cc = np.mean([r['per_subject'][s]['cop'] for r in res['clean']]) * 100
        print(f"{s:>5} {rb:>7.1f}% {rc:>7.1f}% {rc-rb:>+6.1f} | "
              f"{cb:>7.1f}% {cc:>7.1f}% {cc-cb:>+6.1f}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()
    run(args.seeds, args.subjects)


if __name__ == "__main__":
    main()
