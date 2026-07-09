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
    return_factors: bool = False,
):
    """Return NEW sim Trajectory objects with per-subject-scaled positions.

    sim_trajs      : the eegk_sim trajectories to scale.
    real_reference : real trajectories used to compute per-subject targets.
                     For within-subject runs pass TRAIN real only (no leakage).
    match          : "radius" (default) | "step".

    Subjects present in sim but absent from real_reference pass through with
    factor 1.0. Input trajectories are not mutated; positions are copied.
    """
    real_by_subj: Dict[str, List[cd.Trajectory]] = {}
    for t in real_reference:
        real_by_subj.setdefault(t.subject_id, []).append(t)
    sim_by_subj: Dict[str, List[cd.Trajectory]] = {}
    for t in sim_trajs:
        sim_by_subj.setdefault(t.subject_id, []).append(t)

    factors: Dict[str, float] = {}
    for subj, ssubj in sim_by_subj.items():
        rsubj = real_by_subj.get(subj)
        factors[subj] = 1.0 if not rsubj else compute_subject_scale(rsubj, ssubj, match)

    scaled: List[cd.Trajectory] = []
    for t in sim_trajs:
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


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Self-test / preview sim-to-real scaling.")
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--match", default="radius", choices=["radius", "step"])
    args = ap.parse_args()

    real = cd.load_source("eegk_real", repo_root=args.repo_root)
    sim = cd.load_source("eegk_sim", repo_root=args.repo_root)

    scaled, factors = scale_sim_to_real(sim, real, match=args.match, return_factors=True)

    print(f"match={args.match}  per-subject scale factors:")
    for s in sorted(factors):
        print(f"  {s}: {factors[s]:.4f}")

    print("\nmatched-moment check (real vs scaled-sim):")
    for s in sorted({t.subject_id for t in sim}):
        rs = [t for t in real if t.subject_id == s]
        ss_ = [t for t in scaled if t.subject_id == s]
        if args.match == "radius":
            rv, sv = _subject_endpoint_radius_median(rs), _subject_endpoint_radius_median(ss_)
        else:
            rv, sv = _subject_step_mean(rs), _subject_step_mean(ss_)
        print(f"  {s}: real={rv:.4f}  scaled_sim={sv:.4f}  ratio={sv/rv:.3f}")

    base_before = cd.baseline_metrics(sim)
    base_after = cd.baseline_metrics(scaled)
    print("\ndirection-invariance check (sim baseline must be unchanged):")
    print(f"  accuracy   before={base_before['accuracy']*100:.2f}%  after={base_after['accuracy']*100:.2f}%")
    print(f"  angle_err  before={base_before['angle_error_deg']:.2f}  after={base_after['angle_error_deg']:.2f}")
