"""
readout_probe.py  (experimental / one-off diagnostic)
=====================================================
The decisive measurement: on the held-out real test split, compare
  (1) raw BCI endpoint argmax
  (2) our copilot (corrective-velocity rollout)          -- current approach
  (3) direct classifier readout, FINAL-tick argmax        -- no control law
  (4) direct classifier readout, belief-AGGREGATED argmax -- mean softmax over ticks
and, on the trials where the classifier DISAGREES with the raw endpoint, report
which one is right more often. That single contrast says whether the control law
is the bottleneck or whether the gentle blend is genuinely adding value.

Uses the exact 75/25 within-subject models the rest of the pipeline trains.
Run:  python experimental/readout_probe.py --seed 0
"""
from __future__ import annotations
import argparse
from collections import defaultdict

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import torch

import copilot_dataset as cd
import copilot_core as core
from closed_loop import train_copilots


def classifier_readout(model, trajs, norm, mode):
    """Run the LSTM as a plain classifier over the RAW trajectory (no copilot in
    the loop). mode='final' -> last-tick argmax; mode='agg' -> argmax of mean softmax."""
    model.eval()
    preds = []
    with torch.no_grad():
        for t in trajs:
            seq = cd.build_features(t.pos, norm, "basic")          # (T,5) raw features
            x = torch.tensor(seq[None], dtype=torch.float32)
            probs = torch.softmax(model(x)[0], dim=-1).numpy()     # (T,8)
            preds.append(int(probs[-1].argmax() if mode == "final"
                             else probs.mean(0).argmax()))
    return np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true",
                    help="metric-safe cleaning (trim leading dwell + per-session rescale). "
                         "Direction-preserving, so the rawBCI column must not move.")
    ap.add_argument("--sim_frac", type=float, default=0.25,
                    help="EEGK-sim fraction in the training blend; 0 = real-only.")
    ap.add_argument("--eval_norm", choices=("test", "train"), default="test",
                    help="which NormStats scale the vel_mag feature at eval time. "
                         "'test' recomputes from the eval split (the house convention, "
                         "kept as default so earlier logs reproduce); 'train' reuses the "
                         "stats the model was TRAINED with, removing the train/eval "
                         "norm-mismatch confound documented at train_copilot.py --fixed_norm. "
                         "The mismatch differs per blend ratio and per clean/raw, so it is "
                         "not a constant offset across configs.")
    args = ap.parse_args()

    real_all = cd.load_source("eegk_real", clean=args.clean)
    splits = cd.split_real(real_all, seed=args.seed)
    mix = "real-only" if args.sim_frac <= 0 else f"{1-args.sim_frac:.0%}/{args.sim_frac:.0%} real/sim"
    print(f"Training within-subject copilots (seed {args.seed}, {mix}, "
          f"{'clean' if args.clean else 'raw'} inputs, eval_norm={args.eval_norm})...")
    models, norm_map = train_copilots(splits, args.seed, sim_frac=args.sim_frac)

    tot = defaultdict(int)
    dis_tot = {"n": 0, "cls_right": 0, "raw_right": 0}
    print("\n" + "=" * 84)
    print(f"{'subj':6}{'N':>6}{'rawBCI':>9}{'copilot':>9}{'cls_fin':>9}{'cls_agg':>9}"
          f"{'disagree':>10}{'cls|dis':>9}{'raw|dis':>9}")
    print("-" * 84)
    for subj in sorted(models):
        test = splits[subj]["test"]
        if not test:
            continue
        # The model was trained with norm_map[subj]; recomputing from `test` gives it
        # a feature scale it never saw. Both readouts below use whichever is selected.
        norm = cd.compute_norm_stats(test) if args.eval_norm == "test" else norm_map[subj]
        labels = np.array([t.target_label for t in test])
        raw = np.array([cd.label_from_position(t.final_pos) for t in test])
        vels = [(t.pos[1:] - t.pos[:-1]).astype(np.float32) for t in test]
        _, _, cop = core.simulate_batch(models[subj], vels, norm, "0.02", "basic", "cpu", "additive")
        cop = np.array(cop)
        cls_f = classifier_readout(models[subj], test, norm, "final")
        cls_a = classifier_readout(models[subj], test, norm, "agg")

        n = len(test)
        acc = lambda p: (p == labels).mean()
        # disagreement analysis uses the aggregated readout vs raw endpoint
        dis = cls_a != raw
        nd = int(dis.sum())
        cls_right = int((cls_a[dis] == labels[dis]).sum())
        raw_right = int((raw[dis] == labels[dis]).sum())

        print(f"{subj:6}{n:>6}{acc(raw)*100:>8.1f}%{acc(cop)*100:>8.1f}%{acc(cls_f)*100:>8.1f}%"
              f"{acc(cls_a)*100:>8.1f}%{nd:>7}({nd/n*100:>3.0f}%){cls_right/max(nd,1)*100:>8.0f}%"
              f"{raw_right/max(nd,1)*100:>8.0f}%")
        tot["n"] += n
        tot["raw"] += int((raw == labels).sum()); tot["cop"] += int((cop == labels).sum())
        tot["clsf"] += int((cls_f == labels).sum()); tot["clsa"] += int((cls_a == labels).sum())
        dis_tot["n"] += nd; dis_tot["cls_right"] += cls_right; dis_tot["raw_right"] += raw_right

    N = tot["n"]
    print("-" * 84)
    print(f"{'ALL':6}{N:>6}{tot['raw']/N*100:>8.1f}%{tot['cop']/N*100:>8.1f}%"
          f"{tot['clsf']/N*100:>8.1f}%{tot['clsa']/N*100:>8.1f}%"
          f"{dis_tot['n']:>7}({dis_tot['n']/N*100:>3.0f}%)"
          f"{dis_tot['cls_right']/max(dis_tot['n'],1)*100:>8.0f}%"
          f"{dis_tot['raw_right']/max(dis_tot['n'],1)*100:>8.0f}%")
    print("=" * 84)
    print("cls_fin / cls_agg = direct classifier readout (NO control law).")
    print("disagree = % of trials where aggregated readout != raw endpoint;")
    print("cls|dis, raw|dis = accuracy ON those disagreement trials. Whichever is higher")
    print("is the one you should trust when they conflict -> tells you whether to commit.")


if __name__ == "__main__":
    main()
