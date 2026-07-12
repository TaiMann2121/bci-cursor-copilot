"""
blend_constructor.py
====================
Build explicit, fixed-budget BLENDS of trajectory sources for the copilot
training-data experiment, and write them to a CSV that train_copilot.py's
`eegk_blend` branch loads via cd.load_csv_file.

Why this exists
---------------
The training-data hypothesis is "adding non-real data to real improves the
copilot." Testing it cleanly requires blends whose composition is EXPLICIT and
reproducible -- the supervisor's ask, and the fix for the report's Figure 2,
where real-budget and blend-type varied together and muddied every verdict.

Design (decided with the plan)
-------------------------------
* Source-agnostic: a blend is a set of {source_name: ratio} weights over any
  number of already-loaded trajectory pools. real+sim today, real+sim+surrogate
  later, with no code change -- the surrogate slots in as another pool.
* FIXED TOTAL BUDGET (primary): pick N total trials; each source contributes
  round(ratio * N). Total held constant as ratios vary -> isolates COMPOSITION,
  which is the question. Additive mode (keep all real, ADD synthetic) is also
  provided for the sparse-regime "does adding data help" test.
* CALLER CALIBRATES: this module combines pools by ratio and nothing else. The
  caller passes already-calibrated sim (scale_sim_to_real + add_dwell from
  sim_scaling, from TRAIN real only) so calibration stays explicit at the call
  site where the train/test split is known. This module never touches leakage.
* WITHIN-SUBJECT: blends are built per subject, so each subject's blend draws
  only from that subject's pools. Ratios apply per subject.

Leakage note
------------
This module samples from whatever pools it is given. For a leakage-clean
within-subject run, the caller must pass TRAIN-only real and sim calibrated from
TRAIN-only real. Passing full pools will leak. The CLI below wires the clean
path (cd.split_real -> train pools) as the default example.

Key uniqueness
--------------
Blend components share subject IDs (and can share trial numbers), so writing
them to one CSV naively would let _dataframe_to_trajectories' groupby MERGE
ticks from different sources into one trajectory. We therefore re-stamp
run_number / trial_number per emitted trajectory into disjoint ranges per
source, preserving (subject_id, target_label, pos, arm_pred) exactly while
guaranteeing each source's trials stay separate on reload.

RUN
---
    # real + scaled+dwell sim, 50/50, fixed budget, within-subject (train-only)
    python blend_constructor.py --sources eegk_real:0.5 eegk_sim:0.5 \
        --mode fixed_budget --out data/blend/real50_sim50.csv

    # sparse-regime additive: keep ~50 real trials/subject, add sim to 50/50
    python blend_constructor.py --sources eegk_real:0.5 eegk_sim:0.5 \
        --mode additive --real_cap 50 --out data/blend/sparse_real_add_sim.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import copilot_dataset as cd

DT = 0.125  # seconds per tick (matches surrogate_constructor / README downsample)


# --------------------------------------------------------------------------- #
# Ratio label helpers (explicit + identifiable in the report)
# --------------------------------------------------------------------------- #
def ratio_label(weights: Dict[str, float]) -> str:
    """Compact, stable label like 'real70_sim30' from {source: ratio}."""
    short = {"eegk_real": "real", "eegk_sim": "sim", "eegk_surrogate": "surr"}
    parts = []
    for name in sorted(weights):
        pct = int(round(weights[name] * 100))
        parts.append(f"{short.get(name, name)}{pct}")
    return "_".join(parts)


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("blend weights must sum to > 0")
    return {k: v / total for k, v in weights.items()}


# --------------------------------------------------------------------------- #
# Per-subject sampling
# --------------------------------------------------------------------------- #
def _by_subject(pool: Sequence[cd.Trajectory]) -> Dict[str, List[cd.Trajectory]]:
    d: Dict[str, List[cd.Trajectory]] = defaultdict(list)
    for t in pool:
        d[t.subject_id].append(t)
    return d


def _sample(pool: List[cd.Trajectory], n: int, rng) -> List[cd.Trajectory]:
    """Sample n trajectories from pool. Without replacement if enough exist,
    otherwise with replacement (and a flag is surfaced by the caller)."""
    if n <= 0 or len(pool) == 0:
        return []
    replace = n > len(pool)
    idx = rng.choice(len(pool), size=n, replace=replace)
    return [pool[i] for i in idx]


def build_blend(
    pools: Dict[str, Sequence[cd.Trajectory]],
    weights: Dict[str, float],
    mode: str = "fixed_budget",
    budget: int = None,
    real_key: str = "eegk_real",
    real_cap: int = None,
    seed: int = 0,
) -> Tuple[List[cd.Trajectory], Dict[str, dict]]:
    """Construct a per-subject blend from named trajectory pools.

    pools    : {source_name: [Trajectory, ...]} (already calibrated by caller).
    weights  : {source_name: ratio}; normalized internally.
    mode     : 'fixed_budget' -> total trials/subject held ~constant at `budget`
                 (default: that subject's real pool size), each source gets
                 round(ratio * budget).
               'additive'     -> keep real (optionally capped at real_cap), then
                 add other sources so the FINAL proportions match `weights`.
    real_cap : (additive only) cap real trials/subject to simulate the sparse
               regime; the synthetic amount is derived from the ratio.

    Returns (blend_trajectories, per_subject_manifest). Trajectories are
    re-keyed so sources stay separate on CSV reload (see module docstring).
    """
    weights = _normalize(weights)
    rng = np.random.default_rng(seed)
    src_by_subj = {name: _by_subject(p) for name, p in pools.items()}
    subjects = sorted(set().union(*[set(d) for d in src_by_subj.values()]))

    blend: List[cd.Trajectory] = []
    manifest: Dict[str, dict] = {}

    for subj in subjects:
        counts: Dict[str, int] = {}

        if mode == "fixed_budget":
            b = budget if budget is not None else len(src_by_subj.get(real_key, {}).get(subj, []))
            for name, w in weights.items():
                counts[name] = int(round(w * b))

        elif mode == "additive":
            real_pool = src_by_subj.get(real_key, {}).get(subj, [])
            n_real = len(real_pool) if real_cap is None else min(real_cap, len(real_pool))
            counts[real_key] = n_real
            w_real = weights[real_key]
            # total implied by keeping n_real at ratio w_real: N = n_real / w_real
            if w_real <= 0:
                raise ValueError("additive mode needs real_key weight > 0")
            n_total = n_real / w_real
            for name, w in weights.items():
                if name == real_key:
                    continue
                counts[name] = int(round(w * n_total))
        else:
            raise ValueError(f"unknown mode {mode!r} (expected fixed_budget|additive)")

        sub_manifest = {"counts": {}, "resampled": {}}
        for name, n in counts.items():
            pool = src_by_subj.get(name, {}).get(subj, [])
            drawn = _sample(list(pool), n, rng)
            sub_manifest["counts"][name] = len(drawn)
            sub_manifest["resampled"][name] = n > len(pool) and n > 0
            # re-key per source so sources never merge on reload
            for j, t in enumerate(drawn):
                blend.append(_rekey(t, source=name, subject=subj, index=j))
        manifest[subj] = sub_manifest

    return blend, manifest


# stable source -> run_number offset, keeps trial keys disjoint across sources
_SOURCE_RUN_BASE = {"eegk_real": 1000, "eegk_sim": 2000, "eegk_surrogate": 3000}


def _rekey(t: cd.Trajectory, source: str, subject: str, index: int) -> cd.Trajectory:
    """Return a copy with a blend-unique key so the CSV round-trips without
    merging trajectories from different sources that share (subj, trial)."""
    base = _SOURCE_RUN_BASE.get(source, 9000)
    new_keys = (subject, 0, base, index + 1, 1)   # (subj, session, run, trial, inner)
    return cd.Trajectory(
        subject_id=t.subject_id,
        target_label=t.target_label,
        pos=t.pos,            # unchanged; already normalized
        arm_pred=t.arm_pred,
        keys=new_keys,
    )


# --------------------------------------------------------------------------- #
# CSV emission (round-trips through cd.load_csv_file)
# --------------------------------------------------------------------------- #
def blend_to_dataframe(blend: Sequence[cd.Trajectory]) -> pd.DataFrame:
    """Serialize blended trajectories to the canonical trajectory-CSV schema.
    Positions are written back in PIXELS (loader divides by RADIUS_PX)."""
    rows = []
    for t in blend:
        subj, sess, run, trial, inner = t.keys
        tx, ty = cd.TARGET_POS[t.target_label]
        px = t.pos * cd.RADIUS_PX
        for k in range(len(t.pos)):
            rows.append({
                "subject_id": subj,
                "session_number": int(sess),
                "run_number": int(run),
                "trial_number": int(trial),
                "inner_trial_number": int(inner),
                "timestamp_seconds": round(k * DT, 3),
                "cursor_pos_x": float(px[k, 0]),
                "cursor_pos_y": float(px[k, 1]),
                "target_label": int(t.target_label),
                "target_pos_x": tx,
                "target_pos_y": ty,
                "arm_prediction_label": int(t.arm_pred[k]),
            })
    return pd.DataFrame(rows)


def write_blend(blend, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    blend_to_dataframe(blend).to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_sources(items: List[str]) -> Dict[str, float]:
    weights = {}
    for it in items:
        if ":" not in it:
            raise ValueError(f"source spec '{it}' must be name:ratio")
        name, r = it.rsplit(":", 1)
        weights[name] = float(r)
    return weights


def main():
    ap = argparse.ArgumentParser(description="Build fixed-budget trajectory blends.")
    ap.add_argument("--sources", nargs="+", required=True,
                    help="source:ratio specs, e.g. eegk_real:0.7 eegk_sim:0.3")
    ap.add_argument("--mode", default="fixed_budget",
                    choices=["fixed_budget", "additive"])
    ap.add_argument("--budget", type=int, default=None,
                    help="(fixed_budget) total trials/subject; default = subject's real pool size")
    ap.add_argument("--real_cap", type=int, default=None,
                    help="(additive) cap real trials/subject for the sparse regime")
    ap.add_argument("--match", default="radius", choices=["radius", "step"],
                    help="sim calibration moment (scale_sim_to_real)")
    ap.add_argument("--no_dwell", action="store_true", help="skip dwell calibration on sim")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--surrogate_csv", default="data/surrogate/surrogate_trajectories.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    weights = _parse_sources(args.sources)
    label = ratio_label(_normalize(weights))
    out = args.out or f"data/blend/{args.mode}_{label}.csv"

    # --- load real, build leakage-clean TRAIN pools (the clean default path) ---
    import sim_scaling as ss
    real_all = cd.load_source("eegk_real", repo_root=args.repo_root)
    splits = cd.split_real(real_all, seed=args.seed)
    train_real = [t for s in splits.values() for t in s["train"]]

    pools: Dict[str, List[cd.Trajectory]] = {"eegk_real": train_real}

    if "eegk_sim" in weights:
        sim_raw = cd.load_source("eegk_sim", repo_root=args.repo_root)
        sim_cal = ss.scale_sim_to_real(sim_raw, train_real, match=args.match)
        if not args.no_dwell:
            sim_cal = ss.add_dwell_to_sim(sim_cal, train_real, seed=args.seed)
        pools["eegk_sim"] = sim_cal

    if "eegk_surrogate" in weights:
        sp = Path(args.repo_root) / args.surrogate_csv
        if not sp.exists():
            raise SystemExit(f"surrogate requested but not found at {sp}; run surrogate_constructor.py")
        pools["eegk_surrogate"] = cd.load_csv_file(str(sp))

    blend, manifest = build_blend(
        pools, weights, mode=args.mode, budget=args.budget,
        real_cap=args.real_cap, seed=args.seed,
    )
    write_blend(blend, out)

    # report
    print(f"blend '{label}' [{args.mode}] -> {out}")
    print(f"  weights (normalized): { {k: round(v,3) for k,v in _normalize(weights).items()} }")
    total_by_src = defaultdict(int)
    any_resampled = False
    for subj, m in manifest.items():
        for name, c in m["counts"].items():
            total_by_src[name] += c
        for name, r in m["resampled"].items():
            any_resampled = any_resampled or r
    print(f"  trials by source (all subjects): {dict(total_by_src)}")
    print(f"  total trajectories: {sum(total_by_src.values())}")
    if any_resampled:
        print("  NOTE: some source/subject cells were sampled WITH replacement "
              "(requested count exceeded pool size).")
    print(f"  per-subject counts:")
    for subj in sorted(manifest):
        print(f"    {subj}: {manifest[subj]['counts']}")
    print(f"\nNext: python train_copilot.py --training_data eegk_blend "
          f"--blend_csv {out} --train_test within_subject --copilot_vel_mag 0.02")


if __name__ == "__main__":
    main()
