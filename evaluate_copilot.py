"""
evaluate_copilot.py
===================
Authoritative evaluation of a trained copilot run on a held-out eval source
(default: EEGK real). Reads the manifest written by train_copilot.py, maps each
eval subject to the correct model (its own model for within_subject; the single
pooled model otherwise), recomputes normalization from the EVAL source
(deployment-realistic cross-decoder transfer), and reports accuracy + angle error
overall, per subject, and per direction.

RUN
---
    python evaluate_copilot.py --run runs/eegk_sim_sl_within_subject --eval_data eegk_real
    python evaluate_copilot.py --model runs/.../S01/best_model.pt --eval_data eegk_real
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

import copilot_dataset as cd
import copilot_core as core

DEVICE = "cpu"


def load_eval_trajectories(eval_data: str, repo_root: str,
                           eval_split: str = "test", split_seed: int = 0,
                           exclude_calibration: bool = True) -> List[cd.Trajectory]:
    """Trajectories to evaluate on.

    eval_split="test" (default, LEAKAGE-FREE): evaluate ONLY on the held-out test
        blocks from cd.split_real(seed=split_seed) -- the SAME split whose TRAIN
        blocks were used to build the training data. This is the correct held-out
        evaluation. `split_seed` MUST match the seed used at training-data
        construction (blend_constructor / calibration), or train and test will
        not be complementary.
    eval_split="all" (LEGACY, LEAKY for real-trained models): all real minus the
        run-1 calibration block. This overlaps the train blocks and inflates
        results for any model trained on real; kept only for reproducing old
        numbers / debugging.

    For non-real eval sources the split does not apply (returned as-is).
    """
    trajs = cd.load_source(eval_data, repo_root=repo_root)
    if eval_data != "eegk_real":
        return trajs
    if eval_split == "test":
        splits = cd.split_real(trajs, seed=split_seed)
        return [t for sub in splits.values() for t in sub["test"]]
    if eval_split == "all":
        if exclude_calibration:
            trajs = [t for t in trajs if int(t.keys[2]) != 1]
        return trajs
    raise ValueError(f"unknown eval_split {eval_split!r} (expected test|all)")


def load_model(path: str, feature_set: str, hidden_size: int, num_lstm: int) -> core.LSTMCopilot:
    F = 7 if feature_set == "extensive" else 5
    m = core.LSTMCopilot(input_size=F, hidden_size=hidden_size, n_layers=num_lstm)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


def evaluate(run: Optional[str], single_model: Optional[str], eval_data: str,
             vel_mag: str, feature_set: str, hidden_size: int, num_lstm: int,
             repo_root: str, velocity_mode: str = "additive",
             eval_split: str = "test", split_seed: int = 0,
             json_out: Optional[str] = None) -> dict:
    """Evaluate a run/model. Returns a structured results dict (and optionally
    writes it to json_out). Also prints the human-readable report."""
    trajs = load_eval_trajectories(eval_data, repo_root,
                                   eval_split=eval_split, split_seed=split_seed)
    by_subj: Dict[str, List[cd.Trajectory]] = {}
    for t in trajs:
        by_subj.setdefault(t.subject_id, []).append(t)

    # Resolve subject -> model path
    subj_model: Dict[str, str] = {}
    if run:
        manifest = json.loads((Path(run) / "manifest.json").read_text())
        mcfg = manifest["config"]
        vel_mag = mcfg.get("copilot_vel_mag", vel_mag)
        feature_set = mcfg.get("input_feature_set", feature_set)
        hidden_size = mcfg.get("hidden_size", hidden_size)
        num_lstm = mcfg.get("num_lstm", num_lstm)
        velocity_mode = mcfg.get("velocity_mode", velocity_mode)
        models = manifest["models"]
        if "pooled" in models:                      # cross/all_subject: one model for everyone
            for s in by_subj:
                subj_model[s] = models["pooled"]["path"]
        else:                                        # within_subject: per-subject models
            for s in by_subj:
                if s in models:
                    subj_model[s] = models[s]["path"]
                else:
                    print(f"  (no model for {s} in run — skipping)")
    elif single_model:
        for s in by_subj:
            subj_model[s] = single_model
    else:
        raise SystemExit("provide --run <dir> or --model <path>")

    # Cache loaded models
    cache: Dict[str, core.LSTMCopilot] = {}

    def get_model(path):
        if path not in cache:
            cache[path] = load_model(path, feature_set, hidden_size, num_lstm)
        return cache[path]

    per_dir = {i: {"cop": 0, "bci": 0, "n": 0, "cop_err": [], "bci_err": []} for i in range(8)}
    rows = []
    for s in sorted(subj_model):
        strajs = by_subj[s]
        model = get_model(subj_model[s])
        norm = cd.compute_norm_stats(strajs)         # recompute from eval source
        cop = bci = 0
        cerr, berr = [], []
        vels = [(t.pos[1:] - t.pos[:-1]).astype(np.float32) for t in strajs]
        _, finals, preds = core.simulate_batch(model, vels, norm, vel_mag,
                                               feature_set, DEVICE, velocity_mode)
        for t, final, pred in zip(strajs, finals, preds):
            lbl = t.target_label
            cop += int(pred == lbl)
            bci += int(cd.label_from_position(t.final_pos) == lbl)
            ce = cd.angle_error_deg(final, lbl); be = cd.angle_error_deg(t.final_pos, lbl)
            cerr.append(ce); berr.append(be)
            per_dir[lbl]["cop"] += int(pred == lbl)
            per_dir[lbl]["bci"] += int(cd.label_from_position(t.final_pos) == lbl)
            per_dir[lbl]["n"] += 1
            per_dir[lbl]["cop_err"].append(ce); per_dir[lbl]["bci_err"].append(be)
        n = len(strajs)
        rows.append((s, n, bci / n, cop / n, np.mean(berr), np.mean(cerr)))

    # ---- report ----
    print("=" * 70)
    print(f"evaluate_copilot | eval_data={eval_data} | eval_split={eval_split}"
          f"{' seed='+str(split_seed) if eval_split=='test' else ''} | "
          f"{'run='+run if run else 'model='+single_model}")
    print(f"  vel_mag={vel_mag}  mode={velocity_mode}  features={feature_set}  "
          f"hidden={hidden_size}  layers={num_lstm}")
    if eval_split == "all":
        print("  WARNING: eval_split=all is LEAKY for real-trained models "
              "(overlaps train blocks). Use --eval_split test for held-out results.")
    print("=" * 70)
    print(f"  {'Subject':<8}{'N':>6}{'BCI':>9}{'Copilot':>10}{'Δacc':>9}{'BCIerr':>9}{'Coperr':>9}{'Δerr':>8}")
    print("  " + "-" * 66)
    tot_n = tot_bci = tot_cop = 0
    all_berr, all_cerr = [], []
    for s, n, b, c, be, ce in rows:
        print(f"  {s:<8}{n:>6}{b*100:>8.2f}%{c*100:>9.2f}%{(c-b)*100:>+8.2f}"
              f"{be:>8.2f}°{ce:>8.2f}°{be-ce:>+7.2f}°")
        tot_n += n; tot_bci += b * n; tot_cop += c * n
        all_berr += [be] * n; all_cerr += [ce] * n
    B, C = tot_bci / tot_n, tot_cop / tot_n
    print("  " + "-" * 66)
    print(f"  {'OVERALL':<8}{tot_n:>6}{B*100:>8.2f}%{C*100:>9.2f}%{(C-B)*100:>+8.2f}"
          f"{np.mean(all_berr):>8.2f}°{np.mean(all_cerr):>8.2f}°"
          f"{np.mean(all_berr)-np.mean(all_cerr):>+7.2f}°")
    print()
    print("  Per-direction (all subjects):")
    print(f"  {'Dir':<5}{'BCI':>9}{'Copilot':>10}{'Δacc':>9}{'BCIerr':>9}{'Coperr':>9}")
    print("  " + "-" * 50)
    for i in range(8):
        d = per_dir[i]
        if d["n"] == 0:
            continue
        print(f"  {cd.DIR_NAMES[i]:<5}{d['bci']/d['n']*100:>8.1f}%{d['cop']/d['n']*100:>9.1f}%"
              f"{(d['cop']-d['bci'])/d['n']*100:>+8.1f}{np.mean(d['bci_err']):>8.2f}°"
              f"{np.mean(d['cop_err']):>8.2f}°")
    print("=" * 70)

    # ---- structured results (for programmatic use, e.g. the sweep runner) ----
    results = {
        "eval_data": eval_data,
        "eval_split": eval_split,
        "split_seed": split_seed,
        "overall": {
            "n": int(tot_n),
            "bci_acc": float(B),
            "cop_acc": float(C),
            "delta_acc_pp": float((C - B) * 100),
            "bci_err_deg": float(np.mean(all_berr)),
            "cop_err_deg": float(np.mean(all_cerr)),
            "delta_err_deg": float(np.mean(all_berr) - np.mean(all_cerr)),
        },
        "per_subject": {
            s: {
                "n": int(n),
                "bci_acc": float(b), "cop_acc": float(c),
                "delta_acc_pp": float((c - b) * 100),
                "bci_err_deg": float(be), "cop_err_deg": float(ce),
                "delta_err_deg": float(be - ce),
            }
            for (s, n, b, c, be, ce) in rows
        },
    }
    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(results, indent=2))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="training run dir (with manifest.json)")
    ap.add_argument("--model", default=None, help="single model .pt (overrides --run)")
    ap.add_argument("--eval_data", default="eegk_real",
                    choices=["old_decoder", "eegk_sim", "eegk_real"])
    ap.add_argument("--copilot_vel_mag", default="inv_ticks")
    ap.add_argument("--velocity_mode", default="additive",
                    choices=["additive", "combined_fixed"])
    ap.add_argument("--input_feature_set", default="basic", choices=["basic", "extensive"])
    ap.add_argument("--hidden_size", type=int, default=64)
    ap.add_argument("--num_lstm", type=int, default=2)
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--eval_split", default="test", choices=["test", "all"],
                    help="test = held-out blocks (leakage-free, default); "
                         "all = legacy leaky behavior (all real minus run 1)")
    ap.add_argument("--split_seed", type=int, default=0,
                    help="seed for cd.split_real; MUST match training-data construction")
    ap.add_argument("--json_out", default=None, help="optional path to dump results JSON")
    args = ap.parse_args()
    evaluate(args.run, args.model, args.eval_data, args.copilot_vel_mag,
             args.input_feature_set, args.hidden_size, args.num_lstm, args.repo_root,
             args.velocity_mode, eval_split=args.eval_split, split_seed=args.split_seed,
             json_out=args.json_out)


if __name__ == "__main__":
    main()
