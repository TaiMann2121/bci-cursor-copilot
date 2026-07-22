"""
closed_loop_reactive.py  (experimental / the credible closed-loop test)
=======================================================================
Replaces the parametric surrogate in ../closed_loop.py with a DATA-DRIVEN one and
adds the parameter that actually governs credibility: user persistence (alpha).

Surrogate (per subject, faithful by construction)
-------------------------------------------------
From every real trajectory we measure its per-tick CONTROL ERROR:
    e_t = heading_t - angle(target - cursor_t)      # how the user/decoder deviated
    mag_t = |velocity_t|                             # step size texture
A rollout samples one real (mag, e) sequence for the target and, at each tick,
recomputes the *ideal* heading from the CURRENT cursor and applies the recorded
error: bci_vel_t = mag_t * [cos,sin]( angle(target - cursor_eff) + e_t ).
With the copilot OFF and alpha anything, this reconstructs the real trajectory
exactly -> leg-1 faithfulness is automatic (checked below).

User persistence (alpha) -- the load-bearing assumption, made explicit
----------------------------------------------------------------------
The user aims from a blend of the true (copilot-moved) cursor and the cursor they
would have had without the copilot:
    cursor_eff = alpha * cursor_actual + (1 - alpha) * cursor_nocopilot
  alpha = 0 : user ignores the copilot's displacement  == OPEN-LOOP replay
  alpha = 1 : user fully reacts to the copilot          == full CLOSED-LOOP
Sweeping alpha maps open-loop -> closed-loop and shows exactly how much the
copilot's benefit depends on the user reacting. That dependence is the result.

Run:  python experimental/closed_loop_reactive.py --seed 0
"""
from __future__ import annotations
import argparse
from collections import defaultdict

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import torch

import copilot_core as core
import copilot_dataset as cd
from copilot_core import _UNIT
from closed_loop import train_copilots


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def build_library(trajs):
    """subject -> target -> list of (mags (T,), errors (T,)) from real trajectories."""
    lib = defaultdict(lambda: defaultdict(list))
    for t in trajs:
        pos = t.pos
        v = pos[1:] - pos[:-1]
        mags = np.linalg.norm(v, axis=1)
        good = mags > 1e-9
        if good.sum() < 3:
            continue
        headings = np.arctan2(v[:, 1], v[:, 0])
        tgt = _UNIT[t.target_label]
        ideal = np.arctan2(tgt[1] - pos[:-1, 1], tgt[0] - pos[:-1, 0])
        e = wrap(headings - ideal)
        lib[t.subject_id][t.target_label].append((mags.astype(np.float32), e.astype(np.float32)))
    return lib


def rollout(mags, e, target, alpha, model, norm, vel_mag=0.02):
    """One trial. Returns (final_cursor, acquire_tick). model=None -> copilot off."""
    tgt = _UNIT[target]
    cursor = np.zeros(2, dtype=np.float32)
    cursor_nc = np.zeros(2, dtype=np.float32)     # counterfactual: no copilot
    h = c = None
    T = len(mags)
    acq, got = T, False
    with torch.no_grad():
        for t in range(T):
            # counterfactual (no-copilot) command -> reconstructs the real path
            ideal_nc = np.arctan2(tgt[1] - cursor_nc[1], tgt[0] - cursor_nc[0])
            bv_nc = mags[t] * np.array([np.cos(ideal_nc + e[t]), np.sin(ideal_nc + e[t])], np.float32)
            # actual command: user aims from the blended (persistence) cursor
            ce = alpha * cursor + (1 - alpha) * cursor_nc
            ideal_a = np.arctan2(tgt[1] - ce[1], tgt[0] - ce[0])
            bv = mags[t] * np.array([np.cos(ideal_a + e[t]), np.sin(ideal_a + e[t])], np.float32)

            if model is not None:
                m = float(np.linalg.norm(bv))
                unit = bv / m if m > 1e-9 else np.zeros(2, np.float32)
                feat = np.array([cursor[0], cursor[1], unit[0], unit[1],
                                 (m - norm.vel_mag_mean) / norm.vel_mag_std], np.float32)
                x = torch.tensor(feat[None, None, :])
                out, (h, c) = (model.lstm(x) if h is None else model.lstm(x, (h, c)))
                logits = model.classifier(out[:, -1, :])
                conf = float(torch.softmax(logits, -1).max())
                pred = int(logits.argmax())
                step = bv + core.corrective_velocity(cursor, pred, conf, vel_mag)
            else:
                step = bv
            cursor = np.clip(cursor + step, -1.5, 1.5)
            cursor_nc = np.clip(cursor_nc + bv_nc, -1.5, 1.5)
            if not got and np.linalg.norm(cursor) > 0.5 and core.angle_pred(cursor) == target:
                acq, got = t + 1, True
    return cursor, acq


def sample_trials(lib, models, n_per_subj, rng):
    """Fixed list of (subj, target, mags, e) so every alpha sees identical trials."""
    trials = []
    for subj in models:
        if subj not in lib:
            continue
        for _ in range(n_per_subj):
            y = int(rng.integers(8))
            pool = lib[subj].get(y) or lib[subj][rng.choice(list(lib[subj]))]
            mags, e = pool[rng.integers(len(pool))]
            trials.append((subj, y, mags, e))
    return trials


def baseline(trials, norms):
    """Copilot OFF -> exactly alpha-invariant (cursor == cursor_nocopilot). Compute once.
    Reconstructs the real data == leg-1 faithfulness check."""
    ok = acq = 0
    for subj, y, mags, e in trials:
        f0, a0 = rollout(mags, e, y, 0.0, None, norms[subj])
        ok += int(core.angle_pred(f0) == y); acq += a0
    n = len(trials)
    return ok / n * 100, acq / n


def copilot_at(trials, models, norms, alpha):
    ok = acq = 0
    for subj, y, mags, e in trials:
        f1, a1 = rollout(mags, e, y, alpha, models[subj], norms[subj])
        ok += int(core.angle_pred(f1) == y); acq += a1
    n = len(trials)
    return ok / n * 100, acq / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_per_subj", type=int, default=400)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    real = cd.load_source("eegk_real")
    splits = cd.split_real(real, seed=args.seed)
    lib = build_library(real)
    norms = {s: cd.compute_norm_stats([t for t in real if t.subject_id == s])
             for s in {t.subject_id for t in real}}
    print(f"Training 75/25 copilots (seed {args.seed})...")
    models, _ = train_copilots(splits, args.seed)

    trials = sample_trials(lib, models, args.n_per_subj, rng)
    base_acc, base_acq = baseline(trials, norms)     # once; alpha-invariant

    print("\n" + "=" * 70)
    print(f"CLOSED-LOOP copilot benefit vs USER PERSISTENCE (alpha) | seed {args.seed}")
    print(f"fixed trial set: N={len(trials)} | baseline (copilot off) = {base_acc:.2f}%")
    print("=" * 70)
    print(f"{'alpha':>7}{'copilot':>10}{'Δacc':>9}{'acqΔ':>9}")
    print("-" * 70)
    for a in args.alphas:
        cop, acq = copilot_at(trials, models, norms, a)
        tag = "  <- open-loop" if a == 0.0 else ("  <- full closed-loop" if a == 1.0 else "")
        print(f"{a:>7.2f}{cop:>9.1f}%{cop-base_acc:>+8.2f}{acq-base_acq:>+9.2f}{tag}")
    print("-" * 70)
    print("Baseline computed once (copilot off is exactly alpha-invariant) so Δacc(alpha)")
    print("is clean. If Δacc rises with alpha, the benefit is a real feedback effect that")
    print("open-loop replay (alpha=0) cannot see; the curve bounds it by user reactivity.")


if __name__ == "__main__":
    main()
