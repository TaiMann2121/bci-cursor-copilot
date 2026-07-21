"""
surrogate_constructor.py
========================
Builds a per-subject SYNTHETIC trajectory dataset ("EEGK surrogate") calibrated to
each real EEGK subject's decoder profile. The surrogate exists because EEGK real is
eval-only (supervisor's rule), so a training source must be synthetic yet faithful.

Calibration (measured per subject from the real online trajectories, run>=2):
  - step-magnitude pool      : realistic per-tick speeds  -> correct final radius
  - trial-length pool        : realistic tick counts
  - per-tick wobble sigma     : realistic within-trajectory jitter
  - endpoint-error pool       : signed wrapped (final_dir - target_dir), per target
                               -> the calibration target that drives the metric

Generation (per synthetic trial):
  1. pick target y (balanced) and length T (from the pool)
  2. build a wobbly path from measured step magnitudes + tick-wobble
  3. sample an endpoint error e from the subject's real per-target error pool
  4. rigidly ROTATE the whole path so its final direction = angle(y) + e
This reproduces the real endpoint-error distribution exactly while keeping the
per-tick dynamics realistic.

--endpoint_model chooses the calibration source (the notes' open "which parameters"):
  'geometric' (default) : empirical per-target endpoint errors from real online data
  'angle_sigma'         : parametric N(0, angle_sigma_rad) from typing_stats.npz
  'confusion'           : categorical landing via npz arm_confusion + angle_sigma jitter

RUN
---
    python surrogate_constructor.py                       # all subjects, geometric
    python surrogate_constructor.py --endpoint_model angle_sigma --n_per_dir 200
Then:
    python train_copilot.py --training_data eegk_surrogate \
        --surrogate_csv data/surrogate/surrogate_trajectories.csv --train_test within_subject
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# --- archived: put repo root on sys.path so sibling imports still resolve when
#     this script is run from archive/ (see archive/README.md) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import copilot_dataset as cd

EEGK_ROOT = "data/OnlineArmTrajectoryEEGK"
OUT_DIR = "data/surrogate"
DT = 0.125  # seconds per tick (README: 128x downsample of 1024 Hz)


def wrap(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


# --------------------------------------------------------------------------- #
# Measure calibration profile from real online trajectories
# --------------------------------------------------------------------------- #
def measure_profile(subj_trajs: List[cd.Trajectory]) -> dict:
    mags: List[float] = []
    lengths: List[int] = []
    wobble: List[float] = []
    final_radius: List[float] = []
    endpoint_err: Dict[int, List[float]] = defaultdict(list)

    for t in subj_trajs:
        v = t.pos[1:] - t.pos[:-1]
        m = np.linalg.norm(v, axis=1)
        lengths.append(len(v))
        mags.extend(m[m > 1e-6].tolist())
        final_radius.append(float(np.linalg.norm(t.pos[-1])))
        end_ang = np.arctan2(t.pos[-1, 1], t.pos[-1, 0])
        good = m > 1e-6
        if good.any():
            ang = np.arctan2(v[good, 1], v[good, 0])
            wobble.extend(np.abs(wrap(ang - end_ang)).tolist())
        tgt_ang = np.arctan2(cd.UNIT_DIRS[t.target_label, 1], cd.UNIT_DIRS[t.target_label, 0])
        endpoint_err[t.target_label].append(float(wrap(np.array([end_ang - tgt_ang]))[0]))

    return {
        "mags": np.asarray(mags),
        "lengths": np.asarray(lengths),
        "wobble_sigma": float(np.std(wobble)) if wobble else np.radians(20.0),
        "final_radius": np.asarray(final_radius),
        "endpoint_err": {k: np.asarray(v) for k, v in endpoint_err.items()},
    }


# --------------------------------------------------------------------------- #
# Endpoint-error sampler per model
# --------------------------------------------------------------------------- #
def make_endpoint_sampler(profile: dict, subj: str, stats: dict, model: str, rng):
    if model == "geometric":
        pooled = np.concatenate(list(profile["endpoint_err"].values()))

        def sample(y):
            arr = profile["endpoint_err"].get(y)
            if arr is None or len(arr) < 30:
                arr = pooled
            return float(rng.choice(arr))
        return sample

    if model == "angle_sigma":
        sigma = float(stats[subj]["angle_sigma_rad"]) if subj in stats else np.radians(15.0)
        return lambda y: float(rng.normal(0.0, sigma))

    if model == "confusion":
        C = stats[subj]["arm_confusion"] if subj in stats else np.eye(8)
        sigma = float(stats[subj]["angle_sigma_rad"]) if subj in stats else np.radians(15.0)

        def sample(y):
            landed = int(rng.choice(8, p=C[y] / C[y].sum()))
            base = wrap(np.array([np.arctan2(cd.UNIT_DIRS[landed, 1], cd.UNIT_DIRS[landed, 0])
                                  - np.arctan2(cd.UNIT_DIRS[y, 1], cd.UNIT_DIRS[y, 0])]))[0]
            return float(base + rng.normal(0.0, sigma))
        return sample

    raise ValueError(f"unknown endpoint_model {model!r}")


# --------------------------------------------------------------------------- #
# Generate one subject's surrogate trials
# --------------------------------------------------------------------------- #
def generate_subject(subj: str, profile: dict, sample_endpoint, n_per_dir: int,
                     wobble_scale: float, rng) -> pd.DataFrame:
    rows = []
    trial_id = 0
    for y in range(8):
        for _ in range(n_per_dir):
            trial_id += 1
            T = int(rng.choice(profile["lengths"]))
            wob = profile["wobble_sigma"] * wobble_scale
            # wobbly path around a nominal heading of 0
            steps = rng.choice(profile["mags"], size=T)
            angs = rng.normal(0.0, wob, size=T)
            v = np.stack([steps * np.cos(angs), steps * np.sin(angs)], axis=1)
            pos = np.concatenate([np.zeros((1, 2)), np.cumsum(v, axis=0)], axis=0)  # (T+1,2)
            # scale path to a sampled real final radius (keeps temporal texture,
            # fixes the endpoint radius so cursor_pos stays in-distribution)
            r0 = float(np.linalg.norm(pos[-1]))
            rt = float(rng.choice(profile["final_radius"]))
            if r0 > 1e-6:
                pos = pos * (rt / r0)
            # rotate so final direction = angle(y) + e
            e = sample_endpoint(y)
            tgt_ang = np.arctan2(cd.UNIT_DIRS[y, 1], cd.UNIT_DIRS[y, 0])
            phi_des = tgt_ang + e
            phi_raw = np.arctan2(pos[-1, 1], pos[-1, 0])
            dphi = phi_des - phi_raw
            R = np.array([[np.cos(dphi), -np.sin(dphi)], [np.sin(dphi), np.cos(dphi)]])
            pos = pos @ R.T

            px = pos * cd.RADIUS_PX
            tx, ty = cd.TARGET_POS[y]
            for k in range(len(pos)):
                lbl = cd.label_from_position(pos[k]) if np.linalg.norm(pos[k]) > 1e-9 else y
                rows.append({
                    "subject_id": subj, "session_number": 0, "run_number": 2,
                    "trial_number": trial_id, "inner_trial_number": 1,
                    "timestamp_seconds": round(k * DT, 3),
                    "cursor_pos_x": float(px[k, 0]), "cursor_pos_y": float(px[k, 1]),
                    "target_label": y, "target_pos_x": tx, "target_pos_y": ty,
                    "arm_prediction_label": int(lbl),
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint_model", default="geometric",
                    choices=["geometric", "angle_sigma", "confusion"])
    ap.add_argument("--n_per_dir", type=int, default=0,
                    help="trials per direction per subject (0 = match real count)")
    ap.add_argument("--wobble_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--out", default=f"{OUT_DIR}/surrogate_trajectories.csv")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    trajs = [t for t in cd.load_source("eegk_real", repo_root=args.repo_root)
             if int(t.keys[2]) != 1]
    stats = cd.load_typing_stats(str(Path(args.repo_root) / EEGK_ROOT))
    by_subj: Dict[str, List[cd.Trajectory]] = defaultdict(list)
    for t in trajs:
        by_subj[t.subject_id].append(t)

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    all_frames = []
    print(f"Surrogate | endpoint_model={args.endpoint_model} wobble_scale={args.wobble_scale}")
    print(f"{'subj':6}{'n/dir':>7}{'real_acc':>10}{'surr_acc':>10}{'real_err':>10}{'surr_err':>10}")

    for subj in sorted(by_subj):
        strajs = by_subj[subj]
        profile = measure_profile(strajs)
        sampler = make_endpoint_sampler(profile, subj, stats, args.endpoint_model, rng)
        n = args.n_per_dir or max(1, len(strajs) // 8)
        df = generate_subject(subj, profile, sampler, n, args.wobble_scale, rng)
        all_frames.append(df)

        # calibration check: surrogate endpoint profile vs real
        surr = cd._dataframe_to_trajectories(df)
        real_b = cd.baseline_metrics(strajs)
        surr_b = cd.baseline_metrics(surr)
        print(f"{subj:6}{n:>7}{real_b['accuracy']*100:>9.1f}%{surr_b['accuracy']*100:>9.1f}%"
              f"{real_b['angle_error_deg']:>9.1f}°{surr_b['angle_error_deg']:>9.1f}°")

    out = pd.concat(all_frames, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {len(out)} rows ({out.subject_id.nunique()} subjects) -> {args.out}")


if __name__ == "__main__":
    main()
