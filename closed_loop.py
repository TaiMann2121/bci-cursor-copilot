"""
closed_loop.py
==============
A subject-level CLOSED-LOOP surrogate environment for the arm-BCI copilot, plus a
validation harness that quantifies how credible it is.

Why this module exists
----------------------
Every evaluation in this repo so far (evaluate_copilot.py, sweep_blends.py,
trajectory_aware_copilot.py) is OPEN-LOOP: a recorded BCI velocity stream is
replayed and the copilot's correction is added on top. The recorded velocities
never react to where the copilot moves the cursor. But a copilot's entire value
proposition -- the mechanism demonstrated in Lee et al., Nature Machine
Intelligence 2025 (the paper the project brief cites) -- is a *feedback* effect:
an early nudge toward the inferred goal compounds as the user keeps steering
toward the target from the new cursor position. Open-loop replay structurally
cannot reward that, which is why smarter open-loop steering
(trajectory_aware_copilot.py) fails to beat the instantaneous copilot even though
the model's belief points at the true target far more often than the endpoint
lands there.

The closed-loop surrogate is a generative model of the decoder's per-tick output
as noisy, target-directed control FROM THE CURRENT CURSOR:

    intended_dir_t = unit(target_pos - cursor_t)          # proportional control
    heading_t      = angle(intended_dir_t) + N(0, sigma)  # per-subject noise
    bci_vel_t      = step_mag_t * [cos, sin](heading_t)    # real step-mag texture

Because intended_dir_t is recomputed from the *current* cursor, when the copilot
displaces the cursor the surrogate's next command updates -- exactly the feedback
structure a real user provides, and exactly the structure Lee et al. used to train
their copilot in simulation before it transferred to real closed-loop use.

Calibration (per subject, from real EEGK online trajectories only)
------------------------------------------------------------------
  step_mag pool, trial-length pool, onset-dwell pool : copied from real texture
  sigma (per-tick heading noise)                     : the ONE fitted parameter,
      calibrated so the surrogate, run open-loop (no copilot), reproduces that
      subject's real endpoint ACCURACY.

Validation (the credibility argument, made testable)
----------------------------------------------------
  Leg 1 -- open-loop distribution match. Surrogate-vs-real endpoint accuracy is
      matched by construction; angle-error and per-direction accuracy are NOT
      fitted, so their agreement is independent evidence the emulator is faithful.
  Leg 2 -- copilot consistency. The same trained copilot is run open-loop on real
      held-out data and on the surrogate. If its delta-accuracy agrees, the
      surrogate is not handing the copilot artificial structure.
  Leg 3 -- closed-loop payoff. Only after legs 1-2 hold is the closed-loop
      delta interpretable. Reported as delta hit-rate and path efficiency.

STATUS: scaffold. Legs 1-2 are implemented and run below. The closed-loop number
is preliminary (single seed, and it models a target-directed user rather than a
recorded reactive human -- see the "reactive-human" caveat in the progress report).
Do not report the closed-loop delta as a result until it survives multiple seeds
and the leg-1/leg-2 checks pass.

RUN
---
    python closed_loop.py --seed 0 --n_trials 500
"""
from __future__ import annotations

import argparse
import contextlib
import io
import pathlib
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import copilot_core as core
import copilot_dataset as cd
from copilot_core import _UNIT


# --------------------------------------------------------------------------- #
# Per-subject surrogate
# --------------------------------------------------------------------------- #
@dataclass
class SubjectSurrogate:
    subject_id: str
    step_pool: np.ndarray       # per-tick step magnitudes (normalized units)
    length_pool: np.ndarray     # trial lengths (ticks)
    dwell_pool: np.ndarray      # onset dwell (ticks with ~no movement)
    sigma: float                # calibrated per-tick heading noise (radians)
    real_acc: float             # real endpoint accuracy (calibration target)
    real_err: float             # real mean angle error (deg) -- for leg-1 check


def _measure_real(trajs: List[cd.Trajectory]) -> dict:
    """Measure the texture pools + real metrics used for calibration/validation."""
    mags, lengths, dwell = [], [], []
    for t in trajs:
        v = t.pos[1:] - t.pos[:-1]
        m = np.linalg.norm(v, axis=1)
        lengths.append(len(v))
        mags.extend(m[m > 1e-6].tolist())
        # onset dwell = leading ticks whose cumulative displacement stays tiny
        r = np.linalg.norm(t.pos, axis=1)
        moved = np.argmax(r > 0.05) if (r > 0.05).any() else len(r) - 1
        dwell.append(int(moved))
    base = cd.baseline_metrics(trajs)
    return {
        "step_pool": np.asarray(mags) if mags else np.array([0.05]),
        "length_pool": np.asarray(lengths),
        "dwell_pool": np.asarray(dwell),
        "real_acc": base["accuracy"],
        "real_err": base["angle_error_deg"],
    }


def _plan(prof: dict, n: int, rng: np.random.Generator) -> List[dict]:
    """Pre-draw a fixed set of trial plans (target, length, dwell, step mags,
    heading-noise offsets). Replaying the SAME plans with the copilot on vs off
    isolates the copilot's causal effect in closed loop."""
    plans = []
    for i in range(n):
        T = int(rng.choice(prof["length_pool"]))
        d = int(min(rng.choice(prof["dwell_pool"]), max(T - 1, 0)))
        plans.append({
            "target": i % 8,
            "T": T,
            "dwell": d,
            "mags": rng.choice(prof["step_pool"], size=T),
            "noise": rng.standard_normal(T),   # scaled by sigma at rollout time
        })
    return plans


def _rollout(plans: List[dict], sigma: float,
             model: Optional[core.LSTMCopilot], norm: Optional[cd.NormStats],
             vel_mag: float = 0.02) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll the closed-loop surrogate for every plan.

    model=None -> raw BCI (no copilot). Otherwise the instantaneous copilot
    (identical control law to core.simulate_batch) runs in the loop, and because
    the surrogate re-aims from the current cursor each tick, corrections feed back.

    Returns (finals (B,2), preds (B,), acquire_tick (B,)) where acquire_tick is the
    first tick the cursor enters the correct 45-degree wedge at radius>0.5 (T if never).
    """
    B = len(plans)
    finals = np.zeros((B, 2), dtype=np.float32)
    preds = np.zeros(B, dtype=np.int64)
    acquire = np.zeros(B, dtype=np.int64)

    use_copilot = model is not None
    if use_copilot:
        model.eval()

    for i, p in enumerate(plans):
        T, d, tgt = p["T"], p["dwell"], p["target"]
        tgt_pos = _UNIT[tgt]
        cursor = np.zeros(2, dtype=np.float32)
        h = c = None
        acq = T
        got = False
        with torch.no_grad():
            for t in range(T):
                if t < d:
                    bv = np.zeros(2, dtype=np.float32)
                else:
                    to_tgt = tgt_pos - cursor
                    base_ang = np.arctan2(to_tgt[1], to_tgt[0])
                    ang = base_ang + sigma * p["noise"][t]
                    mag = float(p["mags"][t])
                    bv = np.array([mag * np.cos(ang), mag * np.sin(ang)], dtype=np.float32)

                if use_copilot:
                    m = float(np.linalg.norm(bv))
                    unit = bv / m if m > 1e-9 else np.zeros(2, dtype=np.float32)
                    mag_scaled = (m - norm.vel_mag_mean) / norm.vel_mag_std
                    feat = np.array([cursor[0], cursor[1], unit[0], unit[1], mag_scaled],
                                    dtype=np.float32)
                    x = torch.tensor(feat[None, None, :])
                    if h is None:
                        out, (h, c) = model.lstm(x)
                    else:
                        out, (h, c) = model.lstm(x, (h, c))
                    logits = model.classifier(out[:, -1, :])
                    probs = torch.softmax(logits, dim=-1)
                    conf = float(probs.max().item())
                    pred = int(logits.argmax().item())
                    corr = core.corrective_velocity(cursor, pred, conf, vel_mag)
                    step = bv + corr
                else:
                    step = bv

                cursor = np.clip(cursor + step, -1.5, 1.5)
                r = float(np.linalg.norm(cursor))
                if not got and r > 0.5 and core.angle_pred(cursor) == tgt:
                    acq = t + 1
                    got = True

        finals[i] = cursor
        preds[i] = core.angle_pred(cursor)
        acquire[i] = acq
    return finals, preds, acquire


def calibrate_sigma(prof: dict, rng: np.random.Generator,
                    n_cal: int = 400) -> float:
    """Fit the single heading-noise parameter so the surrogate's open-loop
    accuracy matches the subject's real accuracy. Accuracy is monotone in sigma,
    so a coarse grid + local refine suffices."""
    target = prof["real_acc"]

    def acc_at(sigma: float) -> float:
        plans = _plan(prof, n_cal, rng.spawn(1)[0])
        _, preds, _ = _rollout(plans, sigma, None, None)
        labels = np.array([p["target"] for p in plans])
        return float((preds == labels).mean())

    grid = np.linspace(0.1, 1.4, 14)
    accs = np.array([acc_at(s) for s in grid])
    best = grid[int(np.argmin(np.abs(accs - target)))]
    # local refine around the best grid point
    lo, hi = max(0.05, best - 0.1), best + 0.1
    fine = np.linspace(lo, hi, 9)
    faccs = np.array([acc_at(s) for s in fine])
    return float(fine[int(np.argmin(np.abs(faccs - target)))])


def build_surrogates(splits: dict, rng: np.random.Generator) -> Dict[str, SubjectSurrogate]:
    """Calibrate one surrogate per subject from that subject's TRAIN split only."""
    surr = {}
    for subj in sorted(splits):
        train = splits[subj]["train"]
        if len(train) < 20:
            continue
        prof = _measure_real(train)
        sigma = calibrate_sigma(prof, rng)
        surr[subj] = SubjectSurrogate(subj, prof["step_pool"], prof["length_pool"],
                                      prof["dwell_pool"], sigma,
                                      prof["real_acc"], prof["real_err"])
    return surr


# --------------------------------------------------------------------------- #
# Copilot (reuse the exact training path used elsewhere, 75/25 blend)
# --------------------------------------------------------------------------- #
def train_copilots(splits: dict, seed: int) -> Dict[str, core.LSTMCopilot]:
    import blend_constructor as bc
    import sim_scaling as ss
    import train_copilot as tc

    cfg = dict(copilot_vel_mag="0.02", input_feature_set="basic", velocity_mode="additive",
               tick_weighting="exponential", weight_exponent=3.0, num_lstm=2, hidden_size=64,
               learning_rate=1e-3, grad_clip=1.0, batch_size=128, val_frac=0.15)
    real_all = [t for s in splits.values() for t in s["train"]]
    with contextlib.redirect_stderr(io.StringIO()):
        sim = ss.add_dwell_to_sim(ss.scale_sim_to_real(cd.load_source("eegk_sim"), real_all),
                                  real_all, seed=seed)
    pools = {"eegk_real": real_all, "eegk_sim": sim}
    blend, _ = bc.build_blend(pools, {"eegk_real": 0.75, "eegk_sim": 0.25},
                              mode="fixed_budget", budget=None, seed=seed)
    norm_map = {}
    by = defaultdict(list)
    for t in blend:
        by[t.subject_id].append(t)
    models = {}
    for subj, gt in sorted(by.items()):
        norm_map[subj] = cd.compute_norm_stats([t for t in blend if t.subject_id == subj])
        with contextlib.redirect_stdout(io.StringIO()):
            d = pathlib.Path(tempfile.mkdtemp())
            tc.train_one_model([tc.trial_view(t) for t in gt], norm_map[subj], cfg, subj, d)
            m = core.LSTMCopilot(input_size=5, hidden_size=64, n_layers=2)
            m.load_state_dict(torch.load(d / "best_model.pt"))
            m.eval()
        models[subj] = m
    return models, norm_map


# --------------------------------------------------------------------------- #
# Main: run the three validation legs
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_trials", type=int, default=500)
    ap.add_argument("--repo_root", default=".")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    real_all = cd.load_source("eegk_real", repo_root=args.repo_root)
    splits = cd.split_real(real_all, seed=args.seed)

    print("Calibrating per-subject closed-loop surrogates (fit sigma to real accuracy)...")
    surr = build_surrogates(splits, rng)

    print("Training 75/25 copilots (seed %d)..." % args.seed)
    models, norm_map = train_copilots(splits, args.seed)

    # ---- Leg 1: open-loop distribution match (acc fitted, angle-error is not) ----
    print("\n" + "=" * 78)
    print("LEG 1  Surrogate faithfulness (open-loop, no copilot)")
    print("=" * 78)
    print(f"{'subj':6}{'sigma':>7}{'real_acc':>10}{'surr_acc':>10}{'real_err':>10}{'surr_err':>10}")
    for subj, s in surr.items():
        plans = _plan({"length_pool": s.length_pool, "dwell_pool": s.dwell_pool,
                       "step_pool": s.step_pool}, args.n_trials, rng.spawn(1)[0])
        finals, preds, _ = _rollout(plans, s.sigma, None, None)
        labels = np.array([p["target"] for p in plans])
        acc = float((preds == labels).mean())
        err = float(np.mean([cd.angle_error_deg(finals[i], labels[i]) for i in range(len(labels))]))
        print(f"{subj:6}{s.sigma:>7.3f}{s.real_acc*100:>9.1f}%{acc*100:>9.1f}%"
              f"{s.real_err:>9.1f}°{err:>9.1f}°")

    # ---- Leg 2 + Leg 3: copilot open-loop-on-real vs open-loop-on-surrogate vs closed-loop
    print("\n" + "=" * 78)
    print("LEG 2/3  Copilot delta: open-loop(real) vs open-loop(surrogate) vs CLOSED-loop")
    print("=" * 78)
    print(f"{'subj':6}{'OL_real_d':>11}{'OL_surr_d':>11}{'CL_surr_d':>11}{'CL_acqΔ':>10}")
    agg = defaultdict(float); nsum = 0
    for subj, s in surr.items():
        model = models.get(subj)
        norm = norm_map.get(subj)
        if model is None:
            continue
        # open-loop on real held-out test (the deployment-realistic replay number)
        test = splits[subj]["test"]
        if test:
            rnorm = cd.compute_norm_stats(test)
            vels = [(t.pos[1:] - t.pos[:-1]).astype(np.float32) for t in test]
            labs = np.array([t.target_label for t in test])
            bci = np.array([cd.label_from_position(t.final_pos) for t in test])
            _, _, cop = core.simulate_batch(model, vels, rnorm, "0.02", "basic", "cpu", "additive")
            ol_real = float((np.array(cop) == labs).mean() - (bci == labs).mean())
        else:
            ol_real = float("nan")

        # closed-loop surrogate: same plans, copilot OFF vs ON (feedback active)
        plans = _plan({"length_pool": s.length_pool, "dwell_pool": s.dwell_pool,
                       "step_pool": s.step_pool}, args.n_trials, rng.spawn(1)[0])
        labels = np.array([p["target"] for p in plans])
        f0, p0, a0 = _rollout(plans, s.sigma, None, None)          # raw BCI (no copilot)
        f1, p1, a1 = _rollout(plans, s.sigma, model, norm, 0.02)    # copilot in the loop
        cl_bci = float((p0 == labels).mean())
        cl_cop = float((p1 == labels).mean())
        cl_surr = cl_cop - cl_bci
        acq_delta = float(a1.mean() - a0.mean())   # negative = acquires sooner

        # open-loop on the SAME surrogate (copilot cannot feed back): freeze the
        # no-copilot trajectory and add the correction as a post-hoc replay
        snorm = cd.compute_norm_stats(_surr_to_trajs(f0, plans))  # crude; texture-matched
        vels_s = [np.diff(_traj_from_plan(plans[j], s.sigma), axis=0).astype(np.float32)
                  for j in range(len(plans))]
        _, _, cop_s = core.simulate_batch(model, vels_s, snorm, "0.02", "basic", "cpu", "additive")
        ol_surr = float((np.array(cop_s) == labels).mean()
                        - np.mean([core.angle_pred(_traj_from_plan(plans[j], s.sigma)[-1]) == labels[j]
                                   for j in range(len(plans))]))

        print(f"{subj:6}{ol_real*100:>+10.2f}{ol_surr*100:>+11.2f}{cl_surr*100:>+11.2f}{acq_delta:>+10.2f}")
        n = len(plans)
        agg["ol_real"] += ol_real * n; agg["ol_surr"] += ol_surr * n
        agg["cl_surr"] += cl_surr * n; agg["acq"] += acq_delta * n; nsum += n
    print("-" * 78)
    print(f"{'ALL':6}{agg['ol_real']/nsum*100:>+10.2f}{agg['ol_surr']/nsum*100:>+11.2f}"
          f"{agg['cl_surr']/nsum*100:>+11.2f}{agg['acq']/nsum:>+10.2f}")
    print("=" * 78)
    print("Reading: OL_real ~ OL_surr  => surrogate is faithful (leg 2). CL_surr > OL_*")
    print("=> the copilot's feedback effect only shows up closed-loop (the whole point).")
    print("acqΔ<0 => copilot reaches the target wedge sooner (path/time benefit).")
    print("\nCAVEAT: single seed; surrogate models a target-directed user, not a recorded")
    print("reactive human. Treat CL_surr as a scaffold signal, not a final result.")


def _traj_from_plan(p: dict, sigma: float) -> np.ndarray:
    """Deterministic no-copilot trajectory for a plan (for open-loop-on-surrogate)."""
    cursor = np.zeros(2, dtype=np.float32)
    out = [cursor.copy()]
    tgt_pos = _UNIT[p["target"]]
    for t in range(p["T"]):
        if t < p["dwell"]:
            bv = np.zeros(2, dtype=np.float32)
        else:
            to_tgt = tgt_pos - cursor
            ang = np.arctan2(to_tgt[1], to_tgt[0]) + sigma * p["noise"][t]
            mag = float(p["mags"][t])
            bv = np.array([mag * np.cos(ang), mag * np.sin(ang)], dtype=np.float32)
        cursor = np.clip(cursor + bv, -1.5, 1.5)
        out.append(cursor.copy())
    return np.asarray(out)


def _surr_to_trajs(finals, plans):
    """Minimal shim: wrap surrogate endpoints as Trajectory-likes for norm stats."""
    trajs = []
    for j, p in enumerate(plans):
        pos = _traj_from_plan(p, 0.0)  # heading exact; used only for step-mag norm
        trajs.append(cd.Trajectory(subject_id="surr", target_label=p["target"],
                                   pos=pos, arm_pred=np.zeros(len(pos), dtype=np.int64),
                                   keys=("surr", 0, 2, j, 1)))
    return trajs


if __name__ == "__main__":
    main()
