"""
sim_scaling.py
==============
Per-subject rescaling of eegk_sim trajectories so their kinematic scale matches
each subject's real EEGK trajectories, before eegk_sim is blended into copilot
training data.

Why
---
eegk_sim and eegk_real are NOT on the same scale. In normalized (radius) units
the per-tick velocity magnitude is ~0.040 (real) vs ~0.047 (sim), and per-subject
endpoint radius diverges (e.g. S02 median ~0.52 real vs ~0.63 sim). Because
`build_features` feeds cursor position (pos) straight into the LSTM WITHOUT any
per-blend renormalization -- only the vel_mag channel is renormalized by
`compute_norm_stats` -- a scale mismatch in eegk_sim shows up directly as an
out-of-distribution position channel. That is a plausible mechanism for the
observed +0.00 break-even sim result.

Design decisions (driven by copilot_dataset internals)
------------------------------------------------------
* Operates on `cd.Trajectory` objects, scaling `.pos` in normalized units.
  Trajectory.pos is already divided by RADIUS_PX at load, so NO pixel math here.
* One scalar per subject, applied about the origin. The 8 targets are radial, so
  origin-scaling changes reach magnitude but NOT endpoint direction -- the argmax
  metric (label_from_position) is invariant to it. Scaling cannot manufacture
  accuracy; it only aligns distributions. (Verified in the self-test below.)
* match="radius" (default): match median endpoint radius. This is the primary
  because position is the un-renormalized feature channel.
  match="step": match per-tick velocity-magnitude mean instead. Use if you care
  more about the velocity channel; note compute_norm_stats already re-derives
  vel_mag mean/std per group, so this channel partially self-corrects anyway.
  A single origin-scalar cannot match both moments at once (they need different
  factors), nor can it change vel-mag STD independently -- match="step_full"
  raises with guidance rather than silently doing the wrong thing.
* LEAKAGE: factors are calibration, and copilot_dataset's split_real rule is
  "surrogates must be calibrated from TRAIN only". `scale_sim_to_real` therefore
  takes an explicit `real_reference` list -- pass the per-subject TRAIN real
  trajectories, never the full real set, when scaling for a within-subject run.

No norm recomputation is needed here: train_copilot.py already calls
`cd.compute_norm_stats(gtrajs)` per group, so if you scale the sim trajectories
before that call, the blend's vel_mag norm is recomputed for free.

Integration (within_subject)
-----------------------------
    import sim_scaling as ss
    real = cd.load_source("eegk_real", repo_root=root)
    sim  = cd.load_source("eegk_sim",  repo_root=root)
    splits = cd.split_real(real)                       # per-subject train/val/test
    # scale each subject's sim using that subject's TRAIN real only:
    train_real = [t for s in splits.values() for t in s["train"]]
    sim_scaled, factors = ss.scale_sim_to_real(sim, train_real, match="radius",
                                               return_factors=True)
    # then blend sim_scaled with real train and hand to the trainer.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Sequence

import numpy as np

import copilot_dataset as cd


def _subject_step_mean(trajs: Sequence[cd.Trajectory]) -> float:
    """Mean per-tick velocity magnitude (normalized units) over a trajectory set."""
    mags: List[float] = []
    for t in trajs:
        v = cd.per_tick_velocity(t.pos)[1:]          # drop the zero first tick
        mags.extend(np.linalg.norm(v, axis=1).tolist())
    return float(np.mean(mags)) if mags else float("nan")


def _subject_endpoint_radius_median(trajs: Sequence[cd.Trajectory]) -> float:
    """Median endpoint radius (normalized units) over a trajectory set."""
    r = [float(np.linalg.norm(t.final_pos)) for t in trajs]
    return float(np.median(r)) if r else float("nan")


def compute_subject_scale(
    real_subj: Sequence[cd.Trajectory],
    sim_subj: Sequence[cd.Trajectory],
    match: str = "radius",
) -> float:
    """Scalar to multiply sim .pos (about origin) so a chosen moment matches real."""
    if match == "radius":
        r = _subject_endpoint_radius_median(real_subj)
        s = _subject_endpoint_radius_median(sim_subj)
    elif match == "step":
        r = _subject_step_mean(real_subj)
        s = _subject_step_mean(sim_subj)
    elif match == "step_full":
        raise NotImplementedError(
            "Matching vel-mag mean AND std needs a non-origin-scalar transform "
            "(one scalar scales mean and std by the same ratio). Options: per-trial "
            "time-warping, or whiten in velocity space then re-integrate. Decide "
            "whether std-match is worth the complexity before implementing."
        )
    else:
        raise ValueError(f"unknown match mode {match!r} (expected radius|step|step_full)")
    if not np.isfinite(s) or s == 0:
        raise ValueError("degenerate sim moment; cannot compute scale factor")
    return float(r / s)


def scale_sim_to_real(
    sim_trajs: Sequence[cd.Trajectory],
    real_reference: Sequence[cd.Trajectory],
    match: str = "radius",
    restrict_to_reference: bool = True,
    return_factors: bool = False,
):
    """Return NEW sim Trajectory objects with per-subject-scaled positions.

    sim_trajs      : the eegk_sim trajectories to scale.
    real_reference : real trajectories used to compute per-subject targets.
                     For within-subject runs pass TRAIN real only (no leakage).
    match          : "radius" (default) | "step".
    restrict_to_reference : if True (default), sim subjects with NO real
                     counterpart are DROPPED from the output, and a warning names
                     them. This is the correct behaviour for the blend experiment:
                     a subject we can't scale (e.g. eegk_real lacks S03/S06) has no
                     validated scale, so its unscaled sim would be off-distribution
                     data silently folded into any pooled blend. If False, such
                     subjects pass through unscaled (factor 1.0) and are still
                     warned about -- use only if you deliberately want them.

    Input trajectories are not mutated; positions are copied. A returned factors
    dict includes only the subjects actually present in the output.
    """
    real_by_subj: Dict[str, List[cd.Trajectory]] = {}
    for t in real_reference:
        real_by_subj.setdefault(t.subject_id, []).append(t)
    sim_by_subj: Dict[str, List[cd.Trajectory]] = {}
    for t in sim_trajs:
        sim_by_subj.setdefault(t.subject_id, []).append(t)

    unreferenced = sorted(s for s in sim_by_subj if s not in real_by_subj)
    if unreferenced:
        action = "DROPPED" if restrict_to_reference else "kept UNSCALED"
        warnings.warn(
            f"sim subjects with no real reference {action}: {unreferenced} "
            f"(no real data to calibrate a scale factor against).",
            stacklevel=2,
        )

    factors: Dict[str, float] = {}
    for subj, ssubj in sim_by_subj.items():
        rsubj = real_by_subj.get(subj)
        if rsubj:
            factors[subj] = compute_subject_scale(rsubj, ssubj, match)
        elif not restrict_to_reference:
            factors[subj] = 1.0
        # else: leave subj out of factors entirely -> dropped below

    scaled: List[cd.Trajectory] = []
    for t in sim_trajs:
        if t.subject_id not in factors:
            continue                                  # dropped unreferenced subject
        f = factors[t.subject_id]
        scaled.append(
            cd.Trajectory(
                subject_id=t.subject_id,
                target_label=t.target_label,
                pos=t.pos * f,                        # copy; origin-scaled
                arm_pred=t.arm_pred,
                keys=t.keys,
            )
        )
    if return_factors:
        return scaled, factors
    return scaled


# --------------------------------------------------------------------------- #
# Dwell calibration (Option A: prepend a hold phase, preserve sim's movement)
# --------------------------------------------------------------------------- #
# Real EEGK trajectories hold at the origin for a variable "thinking" phase
# (~40-58% of ticks per subject) before the cursor moves, because the decoder
# needs time to produce reliable movement. Sim starts moving at tick ~1. The
# LSTM reads this temporal structure directly, so sim's missing dwell is a
# mismatch that radius scaling does not touch.
#
# Option A (chosen): keep sim's decoded movement EXACTLY, and PREPEND N near-
# origin ticks, with N sampled per subject from the real dwell distribution.
# This aligns the global temporal envelope (adds the hold real has) while
# preserving local texture (sim's actual movement is untouched) -- the same
# "align distribution, don't rewrite texture" rule as scaling. The cost is that
# sim trajectories become ~dwell+13 ticks, slightly longer than real's dominant
# 17; we accept that over time-warping sim's movement to hit a length target.
#
# NOTE on vel_mag: prepending ticks increases T, so the `inv_ticks` copilot
# magnitude (1/T) would shrink. This does NOT affect runs using a fixed
# --copilot_vel_mag (e.g. 0.02); only matters if you switch to inv_ticks.

DWELL_RADIUS = 0.05   # a tick with radius < this counts as "at origin" (dwell)


def measure_dwell_pool(real_reference: Sequence[cd.Trajectory]) -> Dict[str, np.ndarray]:
    """Per-subject array of real dwell lengths (leading near-origin tick counts).

    Derived from real only -- pass TRAIN real for a leakage-clean within-subject
    run, exactly as with the scale factors.
    """
    pools: Dict[str, List[int]] = {}
    for t in real_reference:
        r = np.linalg.norm(t.pos, axis=1)
        k = 0
        while k < len(r) and r[k] < DWELL_RADIUS:
            k += 1
        pools.setdefault(t.subject_id, []).append(k)
    return {s: np.asarray(v) for s, v in pools.items()}


def _sim_lead_dwell(t: cd.Trajectory) -> int:
    """How many leading near-origin ticks sim already has (usually ~1)."""
    r = np.linalg.norm(t.pos, axis=1)
    k = 0
    while k < len(r) and r[k] < DWELL_RADIUS:
        k += 1
    return k


def add_dwell_to_sim(
    sim_trajs: Sequence[cd.Trajectory],
    real_reference: Sequence[cd.Trajectory],
    seed: int = 0,
    return_added: bool = False,
):
    """Prepend a sampled real dwell phase to each sim trajectory (Option A).

    For each sim trajectory, sample a target dwell length from that subject's
    real dwell pool and prepend enough origin ticks to reach it (accounting for
    the ~1 dwell tick sim already has). Sim's movement ticks are copied verbatim.

    sim_trajs      : sim trajectories (typically already radius-scaled).
    real_reference : real trajectories for the dwell pool. TRAIN real only for a
                     leakage-clean within-subject run.
    Subjects with no real reference are passed through unchanged (and warned);
    normally you will have already dropped them via scale_sim_to_real.
    """
    rng = np.random.default_rng(seed)
    pools = measure_dwell_pool(real_reference)

    missing = sorted({t.subject_id for t in sim_trajs} - set(pools))
    if missing:
        warnings.warn(
            f"no real dwell pool for {missing}; those sim trajectories left "
            f"unchanged (dwell not added).",
            stacklevel=2,
        )

    out: List[cd.Trajectory] = []
    added_counts: Dict[str, List[int]] = {}
    for t in sim_trajs:
        pool = pools.get(t.subject_id)
        if pool is None or len(pool) == 0:
            out.append(t)
            continue
        target_dwell = int(rng.choice(pool))
        have = _sim_lead_dwell(t)
        n_add = max(0, target_dwell - have)
        added_counts.setdefault(t.subject_id, []).append(n_add)
        if n_add == 0:
            out.append(t)
            continue
        pad_pos = np.zeros((n_add, 2), dtype=t.pos.dtype)          # held at origin
        pad_pred = np.zeros(n_add, dtype=t.arm_pred.dtype)         # real dwell arm_pred is ~always 0
        out.append(
            cd.Trajectory(
                subject_id=t.subject_id,
                target_label=t.target_label,
                pos=np.vstack([pad_pos, t.pos]),
                arm_pred=np.concatenate([pad_pred, t.arm_pred]),
                keys=t.keys,
            )
        )
    if return_added:
        return out, {s: np.asarray(v) for s, v in added_counts.items()}
    return out


def calibrate_sim(
    sim_trajs: Sequence[cd.Trajectory],
    real_reference: Sequence[cd.Trajectory],
    match: str = "radius",
    add_dwell: bool = True,
    seed: int = 0,
):
    """Full sim calibration: radius/step scaling, then (optionally) dwell prepend.

    Convenience wrapper composing scale_sim_to_real + add_dwell_to_sim with a
    single real_reference. Order matters: scale first (drops unreferenced
    subjects), then add dwell to the survivors.
    """
    scaled = scale_sim_to_real(sim_trajs, real_reference, match=match)
    if add_dwell:
        return add_dwell_to_sim(scaled, real_reference, seed=seed)
    return scaled


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Self-test / preview sim-to-real scaling.")
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--match", default="radius", choices=["radius", "step"])
    args = ap.parse_args()

    real = cd.load_source("eegk_real", repo_root=args.repo_root)
    sim = cd.load_source("eegk_sim", repo_root=args.repo_root)

    scaled, factors = scale_sim_to_real(sim, real, match=args.match, return_factors=True)
    kept = sorted(factors)

    print(f"match={args.match}  per-subject scale factors ({len(kept)} subjects kept):")
    for s in kept:
        print(f"  {s}: {factors[s]:.4f}")

    print("\nmatched-moment check (real vs scaled-sim, kept subjects only):")
    for s in kept:
        rs = [t for t in real if t.subject_id == s]
        ss_ = [t for t in scaled if t.subject_id == s]
        if args.match == "radius":
            rv, sv = _subject_endpoint_radius_median(rs), _subject_endpoint_radius_median(ss_)
        else:
            rv, sv = _subject_step_mean(rs), _subject_step_mean(ss_)
        print(f"  {s}: real={rv:.4f}  scaled_sim={sv:.4f}  ratio={sv/rv:.3f}")

    # baseline must be compared on the SAME subject set that survives scaling
    sim_kept = [t for t in sim if t.subject_id in factors]
    base_before = cd.baseline_metrics(sim_kept)
    base_after = cd.baseline_metrics(scaled)
    print("\ndirection-invariance check (baseline over kept subjects; must be unchanged):")
    print(f"  accuracy   before={base_before['accuracy']*100:.2f}%  after={base_after['accuracy']*100:.2f}%")
    print(f"  angle_err  before={base_before['angle_error_deg']:.2f}  after={base_after['angle_error_deg']:.2f}")

    # --- dwell calibration check ---
    dwelled = add_dwell_to_sim(scaled, real, seed=0)
    print("\ndwell calibration (Option A: prepend real-sampled hold, movement preserved):")
    print(f"  {'subj':5}{'real_dwell_med':>16}{'sim_dwell_after':>18}{'real_len_med':>14}{'sim_len_after':>15}")
    for s in kept:
        rs = [t for t in real if t.subject_id == s]
        ds = [t for t in dwelled if t.subject_id == s]
        rdw = np.median([_sim_lead_dwell(t) for t in rs])
        sdw = np.median([_sim_lead_dwell(t) for t in ds])
        rlen = np.median([t.n_ticks for t in rs])
        slen = np.median([t.n_ticks for t in ds])
        print(f"  {s:5}{rdw:>16.0f}{sdw:>18.0f}{rlen:>14.0f}{slen:>15.0f}")

    # dwell must NOT change endpoint direction either (movement untouched)
    base_dwell = cd.baseline_metrics(dwelled)
    print("\n  direction-invariance after dwell (movement preserved, so unchanged):")
    print(f"    accuracy  scaled={base_after['accuracy']*100:.2f}%  +dwell={base_dwell['accuracy']*100:.2f}%")
    print(f"    angle_err scaled={base_after['angle_error_deg']:.2f}  +dwell={base_dwell['angle_error_deg']:.2f}")
