"""
diagnose_residuals.py
=====================
Decide whether a texture/structure-matched SURROGATE is worth building, by
diagnosing WHERE the real+sim blend still fails on held-out real data.

Logic (from the sim result): sim already carries real EEG *texture*; its only
possible weakness vs a surrogate is *exact global structure* (dwell/length/chaos
per subject). So the surrogate is worth exploring only if the errors sim leaves
behind are the STRUCTURED (chaotic/high-reversal) trials -- the ones a surrogate
with exact chaos-modeling could target. If sim's residual errors are just ordinary
hard trials, no surrogate helps.

For each held-out test trial (leakage-free, split_real(seed)):
  - bci_correct   : raw BCI endpoint classifies correctly
  - real_correct  : 100/0 (real-only) copilot correct
  - blend_correct : 75/25 (real+sim) copilot correct
  - wander        : path_length / net_displacement  (chaos index; high = chaotic)

Then it answers three decision questions:
  Q1 HEADROOM  : of BCI-wrong (correctable) trials, what frac does the blend fix?
  Q2 SIM'S EDGE: trials the blend fixes that real-only can't -- are they MORE
                 chaotic than the correctable pool? (sim closing a structure gap)
  Q3 RESIDUAL  : trials still wrong after blend -- are THEY the chaotic ones?
                 (structure gap remains -> surrogate spec) or ordinary-hard
                 (no lever -> surrogate won't help)
"""
from __future__ import annotations
import argparse, io, contextlib
from collections import defaultdict
import numpy as np

import copilot_dataset as cd
import copilot_core as core
import sim_scaling as ss
import blend_constructor as bc
import train_copilot as tc


def wander_index(pos: np.ndarray) -> float:
    steps = pos[1:] - pos[:-1]
    path = float(np.linalg.norm(steps, axis=1).sum())
    net = float(np.linalg.norm(pos[-1] - pos[0]))
    return path / max(net, 1e-6)


def train_condition(pools, weights, fixed_norm_map, seed, cfg):
    """Build the blend, train one model per subject, return {subj: model}."""
    blend, _ = bc.build_blend({k: pools[k] for k in weights if weights[k] > 0},
                              {k: v for k, v in weights.items() if v > 0},
                              mode="fixed_budget", budget=None, seed=seed)
    groups = defaultdict(list)
    for t in blend:
        groups[t.subject_id].append(t)
    models = {}
    for subj, gtrajs in sorted(groups.items()):
        norm = fixed_norm_map[subj]
        views = [tc.trial_view(t) for t in gtrajs]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            import tempfile, pathlib
            d = pathlib.Path(tempfile.mkdtemp())
            tc.train_one_model(views, norm, cfg, subj, d)
            m = core.LSTMCopilot(input_size=5, hidden_size=64, n_layers=2)
            import torch
            m.load_state_dict(torch.load(d / "best_model.pt"))
            m.eval()
        models[subj] = m
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--chaos_pct", type=float, default=75,
                    help="percentile of wander index above which a trial is 'chaotic'")
    args = ap.parse_args()

    cfg = dict(copilot_vel_mag="0.02", input_feature_set="basic", velocity_mode="additive",
               tick_weighting="exponential", weight_exponent=3.0, num_lstm=2, hidden_size=64,
               learning_rate=1e-3, grad_clip=1.0, batch_size=128, val_frac=0.15)

    # data + leakage-clean split
    real_all = cd.load_source("eegk_real", repo_root=args.repo_root)
    splits = cd.split_real(real_all, seed=args.seed)
    train_real = [t for s in splits.values() for t in s["train"]]
    sim_cal = ss.add_dwell_to_sim(ss.scale_sim_to_real(cd.load_source("eegk_sim", args.repo_root),
                                                       train_real), train_real, seed=args.seed)
    pools = {"eegk_real": train_real, "eegk_sim": sim_cal}

    # fixed norm (real-derived per subject), matching the sweep
    fixed_norm_map = {}
    _by = defaultdict(list)
    for t in real_all:
        _by[t.subject_id].append(t)
    for s, ts in _by.items():
        fixed_norm_map[s] = cd.compute_norm_stats(ts)

    print(f"training real-only (100/0) and blend (75/25), seed={args.seed} ...")
    real_models = train_condition(pools, {"eegk_real": 1.0, "eegk_sim": 0.0},
                                  fixed_norm_map, args.seed, cfg)
    blend_models = train_condition(pools, {"eegk_real": 0.75, "eegk_sim": 0.25},
                                   fixed_norm_map, args.seed, cfg)

    # roll both on held-out test, collect per-trial records
    rows = []  # (subj, bci_ok, real_ok, blend_ok, wander)
    for subj in sorted(splits):
        test = splits[subj]["test"]
        if not test:
            continue
        eval_norm = cd.compute_norm_stats(test)         # as evaluate_copilot does
        vels = [(t.pos[1:] - t.pos[:-1]).astype(np.float32) for t in test]
        _, _, rp = core.simulate_batch(real_models[subj], vels, eval_norm, "0.02", "basic", "cpu", "additive")
        _, _, bp = core.simulate_batch(blend_models[subj], vels, eval_norm, "0.02", "basic", "cpu", "additive")
        for t, rpred, bpred in zip(test, rp, bp):
            lbl = t.target_label
            rows.append((subj,
                         int(cd.label_from_position(t.final_pos) == lbl),
                         int(rpred == lbl), int(bpred == lbl),
                         wander_index(t.pos)))

    subj_arr = np.array([r[0] for r in rows])
    bci = np.array([r[1] for r in rows]); real = np.array([r[2] for r in rows])
    blend = np.array([r[3] for r in rows]); wander = np.array([r[4] for r in rows])
    chaos_thr = np.percentile(wander, args.chaos_pct)
    chaotic = wander >= chaos_thr

    N = len(rows)
    correctable = bci == 0                              # BCI got it wrong
    print("\n" + "=" * 70)
    print(f"RESIDUAL-ERROR DIAGNOSTIC  (seed {args.seed}, N={N} test trials)")
    print(f"chaos threshold: wander >= {chaos_thr:.2f} (top {100-args.chaos_pct:.0f}%)")
    print("=" * 70)
    print(f"raw BCI correct: {bci.mean():.1%} | real-only: {real.mean():.1%} | blend 75/25: {blend.mean():.1%}")

    # Q1 headroom
    fixed_real = correctable & (real == 1)
    fixed_blend = correctable & (blend == 1)
    print(f"\nQ1 HEADROOM (of {correctable.sum()} BCI-wrong trials):")
    print(f"   real-only fixes {fixed_real.sum():3d} ({fixed_real.sum()/correctable.sum():.1%})")
    print(f"   blend     fixes {fixed_blend.sum():3d} ({fixed_blend.sum()/correctable.sum():.1%})")
    print(f"   still wrong after blend: {(correctable & (blend==0)).sum()} "
          f"({(correctable & (blend==0)).sum()/correctable.sum():.1%})")

    # Q2 sim's marginal edge -- trials blend fixes that real-only misses
    blend_only = correctable & (blend == 1) & (real == 0)
    real_only_fix = correctable & (real == 1) & (blend == 0)
    print(f"\nQ2 SIM'S MARGINAL EDGE:")
    print(f"   blend fixes but real-only misses : {blend_only.sum()} trials")
    print(f"   real-only fixes but blend misses : {real_only_fix.sum()} trials  (sim's cost)")
    if blend_only.sum() > 0:
        frac_chaotic_edge = chaotic[blend_only].mean()
        frac_chaotic_pool = chaotic[correctable].mean()
        print(f"   chaotic fraction among sim's fixes : {frac_chaotic_edge:.1%}")
        print(f"   chaotic fraction among correctable : {frac_chaotic_pool:.1%}  (baseline)")
        print(f"   -> sim's fixes are {'MORE' if frac_chaotic_edge>frac_chaotic_pool else 'NOT more'} "
              f"chaotic than the pool "
              f"({'structure gap being closed' if frac_chaotic_edge>frac_chaotic_pool else 'no structure signal'})")

    # Q3 residual character -- still-wrong-after-blend: chaotic or ordinary?
    resid = correctable & (blend == 0)
    print(f"\nQ3 RESIDUAL CHARACTER (of {resid.sum()} still-wrong-after-blend):")
    if resid.sum() > 0:
        frac_chaotic_resid = chaotic[resid].mean()
        base = chaotic[correctable].mean()
        print(f"   chaotic fraction among residuals   : {frac_chaotic_resid:.1%}")
        print(f"   chaotic fraction among correctable : {base:.1%}  (baseline)")
        verdict = ("residuals ENRICHED in chaotic trials -> structure gap remains -> "
                   "surrogate (exact chaos modeling) worth exploring"
                   if frac_chaotic_resid > base + 0.05 else
                   "residuals NOT enriched in chaos -> ordinary-hard trials -> "
                   "surrogate unlikely to help")
        print(f"   -> {verdict}")

    # per-subject blend_only concentration (is it all S07?)
    print(f"\nPer-subject: sim's marginal fixes (blend-only) and residuals:")
    for s in sorted(set(subj_arr)):
        m = subj_arr == s
        bo = (blend_only & m).sum(); rs = (resid & m).sum(); corr = (correctable & m).sum()
        print(f"   {s}: correctable={corr:3d}  sim-only-fixes={bo:2d}  still-wrong={rs:3d}")
    print("=" * 70)


if __name__ == "__main__":
    main()
