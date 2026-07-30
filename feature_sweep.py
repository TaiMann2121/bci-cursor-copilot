"""
feature_sweep.py
================
Item 3 from the 7/27 meeting: tune the feature space we feed the copilot.

Now that diagnose_learning.py has shown the LSTM is functionally sound and there
IS learnable target signal in the trajectory, this asks *which trajectory
information carries that signal*. It trains the same plain 8-way classifier
(no copilot control law) on CLEANED data under different feature-group
combinations and compares validation CE / accuracy.

The central question this is built to answer:
  Is the model just integrating POSITION (which the endpoint metric already
  reads off directly), or do VELOCITY texture / TIMING / directional-CONSISTENCY
  features add decodable target information beyond raw position?

Feature groups (all per-tick):
  pos     cursor_x, cursor_y                      (2)  accumulated position
  vunit   unit velocity vx, vy                    (2)  instantaneous heading
  vmag    (|v| - mean)/std                        (1)  speed texture
  radius  |cursor|                                (1)  distance from center
  time    t/T                                     (1)  fraction of trial elapsed
  cumdir  running-mean unit velocity x, y         (2)  net/consistent heading so far

Configs swept (on cleaned data):
  basic            pos+vunit+vmag        (current production 'basic', 5)
  pos_only         pos                   (2)  -> is position all the signal?
  vel_only         vunit+vmag            (3)  -> can velocity alone decode target?
  pos_vunit        pos+vunit             (4)  -> is speed needed?
  basic_time_rad   basic+time+radius     (7)  -> does knowing 'how far in' help?
  basic_cumdir     basic+cumdir          (7)  -> does directional consistency help?

RUN
---
    python feature_sweep.py                  # all subjects, seed 0
    python feature_sweep.py --seeds 0 1 2    # multi-seed on the winner check
    python feature_sweep.py --subjects S04 S05
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import copilot_dataset as cd
from copilot_dataset import per_tick_velocity
from diagnose_learning import (CHANCE_CE, clean_sessions, load_sessions,
                               train_classifier)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# config name -> ordered list of feature groups
CONFIGS: Dict[str, List[str]] = {
    "basic":          ["pos", "vunit", "vmag"],
    "pos_only":       ["pos"],
    "vel_only":       ["vunit", "vmag"],
    "pos_vunit":      ["pos", "vunit"],
    "basic_time_rad": ["pos", "vunit", "vmag", "time", "radius"],
    "basic_cumdir":   ["pos", "vunit", "vmag", "cumdir"],
}
GROUP_DIM = {"pos": 2, "vunit": 2, "vmag": 1, "radius": 1, "time": 1, "cumdir": 2}


def build_flex(pos: np.ndarray, norm: cd.NormStats, groups: List[str]) -> np.ndarray:
    """Assemble a (T, F) per-tick feature matrix from named groups."""
    T = len(pos)
    v = per_tick_velocity(pos)
    mag = np.linalg.norm(v, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(mag[:, None] > 1e-9, v / np.maximum(mag[:, None], 1e-9), 0.0)

    cols: List[np.ndarray] = []
    for g in groups:
        if g == "pos":
            cols += [pos[:, 0], pos[:, 1]]
        elif g == "vunit":
            cols += [unit[:, 0], unit[:, 1]]
        elif g == "vmag":
            cols += [(mag - norm.vel_mag_mean) / norm.vel_mag_std]
        elif g == "radius":
            cols += [np.linalg.norm(pos, axis=1)]
        elif g == "time":
            cols += [np.arange(T, dtype=np.float64) / max(T - 1, 1)]
        elif g == "cumdir":
            csum = np.cumsum(unit, axis=0)
            denom = np.arange(1, T + 1)[:, None]
            mean_head = csum / denom            # running-mean heading (|.|<=1)
            cols += [mean_head[:, 0], mean_head[:, 1]]
        else:
            raise ValueError(f"unknown group {g!r}")
    return np.stack(cols, axis=1).astype(np.float32)


def build_subject(trajs, groups) -> Tuple[List[np.ndarray], List[int]]:
    norm = cd.compute_norm_stats(trajs)
    seqs = [build_flex(t.pos, norm, groups) for t in trajs]
    labels = [t.target_label for t in trajs]
    return seqs, labels


def run(subjects, epochs, seeds, out_dir: Path):
    raw = load_sessions("." + "/" + cd.DATA_PATHS["eegk_real"])
    clean = clean_sessions(raw, do_trim=True, do_scale=True)
    subj_map = defaultdict(list)
    for s in clean:
        if subjects and s.subject_id not in subjects:
            continue
        subj_map[s.subject_id].extend(s.trajs)
    subj_map = dict(sorted(subj_map.items()))

    # results[config][subj] = {"val_ce": mean over seeds, "val_acc": ...}
    results: Dict[str, Dict[str, dict]] = {c: {} for c in CONFIGS}
    for cname, groups in CONFIGS.items():
        F = sum(GROUP_DIM[g] for g in groups)
        for subj, trajs in subj_map.items():
            seqs, labels = build_subject(trajs, groups)
            ces, accs = [], []
            for seed in seeds:
                h = train_classifier(seqs, labels, F, epochs, seed)
                ces.append(h["val_ce"][-1]); accs.append(h["val_acc"][-1])
            results[cname][subj] = {"val_ce": float(np.mean(ces)),
                                    "val_ce_sd": float(np.std(ces)),
                                    "val_acc": float(np.mean(accs)),
                                    "F": F}

    # ---- report ----
    subj_list = list(subj_map)
    print("\n" + "=" * 100)
    print(f"FEATURE-SPACE SWEEP on CLEANED data  (epochs={epochs}, seeds={seeds})")
    print(f"metric = held-out validation CE (lower better); chance = {CHANCE_CE:.3f}")
    print("=" * 100)
    print(f"{'config':>15} {'F':>3} | " + " ".join(f"{s:>7}" for s in subj_list)
          + f" | {'mean_CE':>8} {'mean_acc':>8}")
    print("-" * 100)
    ranking = []
    for cname in CONFIGS:
        F = results[cname][subj_list[0]]["F"]
        ces = [results[cname][s]["val_ce"] for s in subj_list]
        accs = [results[cname][s]["val_acc"] for s in subj_list]
        mce, macc = float(np.mean(ces)), float(np.mean(accs))
        ranking.append((mce, macc, cname))
        print(f"{cname:>15} {F:>3} | " + " ".join(f"{c:>7.3f}" for c in ces)
              + f" | {mce:>8.3f} {macc*100:>7.1f}%")
    print("-" * 100)
    ranking.sort()
    print("ranked by mean val CE (best first):")
    for mce, macc, cname in ranking:
        d = mce - ranking[0][0]
        print(f"   {cname:>15}  CE={mce:.3f} (+{d:.3f})  acc={macc*100:.1f}%")
    best = ranking[0][2]
    base = next(r for r in ranking if r[2] == "basic")
    print(f"\nbest: {best} (CE {ranking[0][0]:.3f})   vs current 'basic' "
          f"(CE {base[0]:.3f}, acc {base[1]*100:.1f}%)")
    print("interpretation:")
    print("  pos_only vs vel_only  -> how much target signal is position vs velocity")
    print("  basic vs pos_vunit    -> whether speed (vmag) adds anything")
    print("  basic_* vs basic      -> whether timing/consistency features add signal")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_sweep.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {out_dir/'feature_sweep.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--out_dir", default="results/feature_sweep")
    args = ap.parse_args()
    run(args.subjects, args.epochs, args.seeds, Path(args.out_dir))


if __name__ == "__main__":
    main()
