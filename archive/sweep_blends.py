"""
sweep_blends.py
===============
Run the fixed-budget real/sim blend-ratio sweep end to end and collect a single
comparison table (Δaccuracy AND Δangle-error, per subject + overall) on the
LEAKAGE-FREE held-out real test split.

What it does, per ratio r in the grid (r = real fraction):
  1. blend_constructor: build a fixed-budget blend from the TRAIN split
     (real:r, sim:1-r), same --seed for every ratio -> identical split/test set.
  2. train_copilot: train within-subject copilots on that blend, into a
     ratio-specific run dir (so ratios don't overwrite each other).
  3. evaluate_copilot: evaluate on the held-out TEST split (eval_split=test,
     matching --seed) -> returns Δacc / Δerr. NO leakage.

The 100/0 row (real only, from the train split) is the honest baseline every
other ratio is judged against. Because the real-only baseline is built through
the blend constructor, it too is trained on the train split and evaluated on
test -- so the baseline is finally leakage-free, unlike prior reports.

RUN
---
    python sweep_blends.py                       # default grid 1.0..0.0, seed 0
    python sweep_blends.py --ratios 1.0 0.5 0.0  # custom grid
    python sweep_blends.py --budget 200          # fixed budget/subject
    python sweep_blends.py --dry_run             # print the plan, run nothing

Notes
-----
* Requires blend_constructor.py, train_copilot.py, evaluate_copilot.py, and the
  fixed (test-split) evaluate. Train is run as a subprocess (heavy); blend build
  and eval are imported (light, return values).
* --seed is shared across every stage so the train/test partition is IDENTICAL
  for all ratios -- the whole point, for comparability.
* Trajectory count per ratio is capped by --budget (default: each subject's
  train pool size). A ratio that requests more sim than exists resamples with
  replacement (blend_constructor warns); watch for that on sparse subjects.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# --- archived: put repo root on sys.path so sibling imports still resolve when
#     this script is run from archive/ (see archive/README.md) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import copilot_dataset as cd
import sim_scaling as ss
import blend_constructor as bc
import evaluate_copilot as ev


def ratio_tag(r_real: float, mode: str = "fixed_budget") -> str:
    if mode == "additive":
        # additive keeps ALL real and ADDS sim = (1-r)/r of it. Label by % added.
        pct_added = int(round((1.0 - r_real) / r_real * 100)) if r_real > 0 else 0
        return "real100" if pct_added == 0 else f"real100_add{pct_added}sim"
    return f"real{int(round(r_real*100))}_sim{int(round((1-r_real)*100))}"


def build_pools(repo_root: str, seed: int, match: str, add_dwell: bool):
    """Leakage-clean TRAIN pools: real train split + sim calibrated from it."""
    real_all = cd.load_source("eegk_real", repo_root=repo_root)
    splits = cd.split_real(real_all, seed=seed)
    train_real = [t for s in splits.values() for t in s["train"]]
    sim_raw = cd.load_source("eegk_sim", repo_root=repo_root)
    sim_cal = ss.scale_sim_to_real(sim_raw, train_real, match=match)
    if add_dwell:
        sim_cal = ss.add_dwell_to_sim(sim_cal, train_real, seed=seed)
    return {"eegk_real": train_real, "eegk_sim": sim_cal}


def run_one_ratio(r_real: float, pools, args) -> dict:
    tag = ratio_tag(r_real, args.mode)
    blend_csv = Path(args.repo_root) / "data/blend" / f"sweep_{tag}.csv"
    run_root = Path(args.repo_root) / "runs/sweep" / tag
    run_dir = run_root / "eegk_blend_sl_within_subject"
    json_out = Path(args.repo_root) / "results/sweep" / f"{tag}.json"

    weights = {"eegk_real": r_real, "eegk_sim": 1.0 - r_real}

    # 1) build blend from TRAIN pools.
    #    fixed_budget: pass ALL pools + keep zero weights so budget sizes to the
    #                  real pool (no oversampling; trades real for sim at fixed total).
    #    additive:     keep ALL real, ADD sim = (1-r)/r x real. More sim = more total.
    blend, manifest = bc.build_blend(
        pools, weights,
        mode=args.mode, budget=args.budget, seed=args.seed,
    )
    bc.write_blend(blend, str(blend_csv))
    n_total = sum(sum(m["counts"].values()) for m in manifest.values())
    if n_total == 0:
        raise SystemExit(
            f"[{tag}] produced an EMPTY blend (0 trials). Check pools/ratios.")
    # GUARD: additive can request more sim than exists -> sampling with replacement
    # (duplicated trials), which reintroduces the oversampling confound. Fail loud.
    resampled = [s for s, m in manifest.items()
                 if any(m["resampled"].get(k) for k in m["resampled"])]
    if resampled:
        raise SystemExit(
            f"[{tag}] OVERSAMPLING: subjects {resampled} requested more sim than the "
            f"pool holds, so trials would be DUPLICATED (with-replacement sampling). "
            f"This reintroduces the oversampling confound. Reduce the added-sim level "
            f"(keep the grid at r_real >= 0.5, i.e. at most +100% sim) so every "
            f"subject stays within its sim pool.")
    print(f"  [{tag}] blend built: {n_total} trials -> {blend_csv}")

    # 2) train (subprocess; heavy)
    train_cmd = [
        sys.executable, "train_copilot.py",
        "--training_data", "eegk_blend",
        "--blend_csv", str(blend_csv),
        "--train_test", "within_subject",
        "--copilot_vel_mag", str(args.copilot_vel_mag),
        "--out_root", str(run_root),
        "--repo_root", args.repo_root,
    ]
    if args.fixed_norm:
        train_cmd.append("--fixed_norm")
    print(f"  [{tag}] training ...")
    r = subprocess.run(train_cmd, cwd=args.repo_root, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise SystemExit(f"training failed for {tag}")

    # 3) evaluate on held-out TEST split (imported; returns dict)
    res = ev.evaluate(
        run=str(run_dir), single_model=None, eval_data="eegk_real",
        vel_mag=args.copilot_vel_mag, feature_set="basic",
        hidden_size=64, num_lstm=2, repo_root=args.repo_root,
        eval_split="test", split_seed=args.seed, json_out=str(json_out),
    )
    return res


def print_table(grid: List[float], results: Dict[str, dict], mode: str = "fixed_budget"):
    """One comparison table: rows = ratios, Δacc per subject + overall, Δerr overall."""
    subjects = sorted(next(iter(results.values()))["per_subject"].keys())
    base_tag = ratio_tag(grid[0], mode)  # first ratio = baseline (real only)

    hdr_mode = ("ADDITIVE (keep all real, ADD sim)" if mode == "additive"
                else "FIXED BUDGET (equal total)")
    print("\n" + "=" * 84)
    print(f"BLEND SWEEP  [{hdr_mode}]  held-out real test split, leakage-free")
    print("=" * 84)
    hdr = f"{'condition':<22}" + "".join(f"{s:>8}" for s in subjects) + f"{'OVERALL':>10}{'Δerr°':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in grid:
        tag = ratio_tag(r, mode)
        res = results[tag]
        cells = "".join(f"{res['per_subject'][s]['delta_acc_pp']:>+8.2f}" for s in subjects)
        ov = res["overall"]["delta_acc_pp"]
        de = res["overall"]["delta_err_deg"]
        mark = "  <- baseline" if tag == base_tag else ""
        print(f"{tag:<22}{cells}{ov:>+10.2f}{de:>+9.2f}{mark}")
    print("-" * len(hdr))
    print("Δacc = copilot − raw BCI (pp). Compare every row against the baseline (real-only).")
    if mode == "additive":
        print("ADDITIVE: rows add sim ON TOP of all real, so they use MORE data than baseline")
        print("(tests 'does adding data help'; pair with the fixed-budget run to rule out")
        print("a pure quantity effect).")
    print("=" * 84)


def main():
    ap = argparse.ArgumentParser(description="Fixed-budget real/sim blend-ratio sweep.")
    ap.add_argument("--ratios", nargs="+", type=float, default=None,
                    help="real fractions (first = baseline). Default depends on --mode: "
                         "fixed_budget uses 1.0..0.0; additive uses 1.0,0.75,0.5 "
                         "(baseline, +33%% sim, +100%% sim).")
    ap.add_argument("--budget", type=int, default=None,
                    help="fixed trials PER SUBJECT (default: that subject's real "
                         "train-pool size, size-matched across ratios, no oversampling). "
                         "An explicit value applies to every subject, so keep it <= the "
                         "smallest subject's pool to avoid sampling with replacement.")
    ap.add_argument("--mode", default="fixed_budget",
                    choices=["fixed_budget", "additive"],
                    help="fixed_budget: equal total, trade real for sim (composition). "
                         "additive: keep all real, ADD sim on top (tests 'adding data helps').")
    ap.add_argument("--match", default="radius", choices=["radius", "step"])
    ap.add_argument("--no_dwell", action="store_true")
    ap.add_argument("--copilot_vel_mag", default="0.02")
    ap.add_argument("--seed", type=int, default=0,
                    help="shared seed for split + sampling; identical test set across ratios")
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--fixed_norm", action="store_true",
                    help="use real-derived vel-mag norm for every ratio (isolates "
                         "data composition; removes the train/eval norm confound)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    # Default grid depends on mode. Additive caps at r=0.5 (+100% sim) so no subject
    # oversamples its sim pool; fixed_budget can span the full 1.0..0.0 range.
    if args.ratios is None:
        args.ratios = [1.0, 0.75, 0.5] if args.mode == "additive" else [1.0, 0.75, 0.5, 0.25, 0.0]

    print(f"Sweep plan [{args.mode}]: ratios={args.ratios} seed={args.seed} "
          f"fixed_norm={args.fixed_norm}")
    print(f"  eval: held-out TEST split (leakage-free), split_seed={args.seed}")
    if args.dry_run:
        for r in args.ratios:
            print(f"  would run {ratio_tag(r, args.mode)}")
        return

    pools = build_pools(args.repo_root, args.seed, args.match, not args.no_dwell)
    print(f"  train pools: real={len(pools['eegk_real'])} sim={len(pools['eegk_sim'])}")

    results: Dict[str, dict] = {}
    for r in args.ratios:
        results[ratio_tag(r, args.mode)] = run_one_ratio(r, pools, args)

    print_table(args.ratios, results, args.mode)


if __name__ == "__main__":
    main()
