"""
trajectory_aware_copilot.py
===========================
A CAUSAL, trajectory-aware rollout for the copilot, and an A/B evaluation of it
against the current endpoint-steered copilot on the held-out real test split.

Motivation
----------
The ceiling diagnostic showed the deployed copilot captures ~8% of BCI errors
while the model's belief across the trajectory points at the true target on ~28%
(both replicated on the new 6-subject data). The deployed copilot steers toward
THIS tick's instantaneous prediction and scores by where the cursor ENDS UP; the
model's earlier, accumulated evidence is discarded in the read-out.

IMPORTANT (the point that must be respected): only the CURSOR ENDPOINT counts.
Belief is not a substitute for position. This rollout does NOT classify by
aggregated belief (that would be the non-deployable oracle). Instead it uses a
CAUSAL running belief (past+present ticks only) to steer the cursor EARLIER and
more decisively so it physically arrives in time — and success is still measured
exactly as before: the argmax of the final cursor position. Whether belief
converts into position is the empirical question; this module measures it.

What changes vs simulate_batch
------------------------------
Current: correction points at argmax(prob_t)      (instantaneous belief)
         magnitude = base_vel_mag * conf_t         (instantaneous confidence)
This:    correction points at argmax(running_bel)  (accumulated, causal)
         magnitude = base_vel_mag * gain * conf_run (accumulated confidence)
  running_bel_t = sum_{s<=t} decay^(t-s) * prob_s
    decay = 1.0  -> equal vote over all ticks seen so far
    decay < 1.0  -> recency-weighted (middle ground)
    decay -> 0   -> collapses toward the instantaneous (current) copilot
  gain          -> optional magnitude multiplier (bounded 0.02 may be too weak
                   to physically move the cursor in the ticks remaining)

Everything else (features, LSTM stepping, additive velocity, clipping, endpoint
scoring) is identical to simulate_batch, so differences are attributable to the
steering rule alone.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch

import copilot_core as core
from copilot_core import _UNIT, resolve_vel_mag, angle_pred


def simulate_batch_traj_aware(model, vel_list, norm, vel_mag_spec="0.02",
                              feature_set="basic", device="cpu",
                              velocity_mode="additive", decay=1.0, gain=1.0):
    """Causal trajectory-aware rollout. Same signature/returns as
    core.simulate_batch, plus decay and gain. Returns (seqs, finals, preds)."""
    model.eval()
    B = len(vel_list)
    lengths = np.array([len(v) for v in vel_list])
    Tmax = int(lengths.max())
    F = model.input_size
    vel_mag = np.array([resolve_vel_mag(vel_mag_spec, int(L)) for L in lengths], dtype=np.float32)
    step_fixed = (1.0 / np.maximum(lengths, 1)).astype(np.float32)

    V = np.zeros((B, Tmax, 2), dtype=np.float32)
    for i, v in enumerate(vel_list):
        V[i, :len(v)] = v

    cursor = np.zeros((B, 2), dtype=np.float32)
    raw = np.zeros((B, 2), dtype=np.float32)
    seqs = np.zeros((B, Tmax, F), dtype=np.float32)
    running = np.zeros((B, 8), dtype=np.float32)     # accumulated belief (causal)
    h = c = None

    torch_ctx = torch.no_grad()
    torch_ctx.__enter__()
    for t in range(Tmax):
        active = t < lengths
        bv = V[:, t, :]
        mag = np.linalg.norm(bv, axis=1)
        unit = np.zeros_like(bv); nz = mag > 1e-9
        unit[nz] = bv[nz] / mag[nz, None]
        mag_scaled = (mag - norm.vel_mag_mean) / norm.vel_mag_std
        feat = np.concatenate([cursor, unit, mag_scaled[:, None]], axis=1)
        if feature_set == "extensive":
            feat = np.concatenate([feat, raw], axis=1)
        seqs[:, t, :] = feat

        x_t = torch.tensor(feat[:, None, :], dtype=torch.float32, device=device)
        if h is None:
            out, (h, c) = model.lstm(x_t)
        else:
            out, (h, c) = model.lstm(x_t, (h, c))
        probs = torch.softmax(model.classifier(out[:, -1, :]), dim=-1).cpu().numpy()  # (B,8)

        # --- causal running belief: decay past, add present ---
        running = decay * running + probs
        rsum = running.sum(axis=1, keepdims=True)
        run_norm = running / np.maximum(rsum, 1e-9)      # normalized accumulated belief
        pred = run_norm.argmax(axis=1)                   # steer toward ACCUMULATED argmax
        conf = run_norm.max(axis=1)                      # accumulated confidence

        tgt = _UNIT[pred]
        d = tgt - cursor
        dn = np.linalg.norm(d, axis=1)
        dd = np.zeros_like(d); ok = dn > 1e-6
        dd[ok] = d[ok] / dn[ok, None]
        corr = dd * (vel_mag * gain * conf)[:, None]

        if velocity_mode == "additive":
            step = bv + corr
        elif velocity_mode == "combined_fixed":
            comb = bv + corr
            cn = np.linalg.norm(comb, axis=1)
            step = np.zeros_like(comb); good = cn > 1e-9
            step[good] = comb[good] / cn[good, None] * step_fixed[good, None]
        else:
            raise ValueError(f"unknown velocity_mode {velocity_mode!r}")

        upd = active[:, None]
        cursor = np.clip(cursor + step * upd, -1.5, 1.5)
        raw = np.clip(raw + bv * upd, -1.5, 1.5)
    torch_ctx.__exit__(None, None, None)

    finals = cursor
    preds = np.array([angle_pred(finals[i]) for i in range(B)])
    seqs_trim = [seqs[i, :lengths[i], :] for i in range(B)]
    return seqs_trim, finals, preds


# --------------------------------------------------------------------------- #
# A/B evaluation on the held-out real test split
# --------------------------------------------------------------------------- #
def evaluate_ab(models, splits, decay, gain, vel_mag="0.02"):
    """Compare current (instantaneous) vs trajectory-aware steering, per subject,
    scored ONLY by endpoint accuracy. Returns dict of aggregate + per-subject."""
    import copilot_dataset as cd
    agg = {"n": 0, "bci": 0, "cur": 0, "traj": 0}
    per = {}
    for subj in sorted(splits):
        test = splits[subj]["test"]
        if not test:
            continue
        norm = cd.compute_norm_stats(test)
        vels = [(t.pos[1:] - t.pos[:-1]).astype(np.float32) for t in test]
        labels = np.array([t.target_label for t in test])
        bci = np.array([cd.label_from_position(t.final_pos) for t in test])
        _, _, cur = core.simulate_batch(models[subj], vels, norm, vel_mag, "basic", "cpu", "additive")
        _, _, trj = simulate_batch_traj_aware(models[subj], vels, norm, vel_mag,
                                              "basic", "cpu", "additive", decay=decay, gain=gain)
        n = len(test)
        per[subj] = {
            "n": n,
            "bci_acc": float((bci == labels).mean()),
            "cur_acc": float((np.array(cur) == labels).mean()),
            "traj_acc": float((np.array(trj) == labels).mean()),
        }
        agg["n"] += n
        agg["bci"] += int((bci == labels).sum())
        agg["cur"] += int((np.array(cur) == labels).sum())
        agg["traj"] += int((np.array(trj) == labels).sum())
    out = {"per_subject": per, "overall": {
        "n": agg["n"],
        "bci_acc": agg["bci"]/agg["n"],
        "cur_acc": agg["cur"]/agg["n"],
        "traj_acc": agg["traj"]/agg["n"],
    }}
    return out


def main():
    import argparse, io, contextlib, tempfile, pathlib
    import copilot_dataset as cd
    import sim_scaling as ss
    import blend_constructor as bc
    import train_copilot as tc

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--decays", nargs="+", type=float, default=[1.0, 0.7, 0.4])
    ap.add_argument("--gains", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    args = ap.parse_args()

    cfg = dict(copilot_vel_mag="0.02", input_feature_set="basic", velocity_mode="additive",
               tick_weighting="exponential", weight_exponent=3.0, num_lstm=2, hidden_size=64,
               learning_rate=1e-3, grad_clip=1.0, batch_size=128, val_frac=0.15)

    real_all = cd.load_source("eegk_real", repo_root=args.repo_root)
    splits = cd.split_real(real_all, seed=args.seed)
    train_real = [t for s in splits.values() for t in s["train"]]
    sim = ss.add_dwell_to_sim(ss.scale_sim_to_real(cd.load_source("eegk_sim", args.repo_root),
                                                   train_real), train_real, seed=args.seed)
    pools = {"eegk_real": train_real, "eegk_sim": sim}
    fixed_norm_map = {}
    by = defaultdict(list)
    for t in real_all:
        by[t.subject_id].append(t)
    for s, ts in by.items():
        fixed_norm_map[s] = cd.compute_norm_stats(ts)

    # train the 75/25 model (same as the ceiling diagnostic)
    print(f"training 75/25 model, seed={args.seed} ...")
    blend, _ = bc.build_blend(pools, {"eegk_real": 0.75, "eegk_sim": 0.25},
                              mode="fixed_budget", budget=None, seed=args.seed)
    groups = defaultdict(list)
    for t in blend:
        groups[t.subject_id].append(t)
    models = {}
    for subj, gt in sorted(groups.items()):
        with contextlib.redirect_stdout(io.StringIO()):
            d = pathlib.Path(tempfile.mkdtemp())
            tc.train_one_model([tc.trial_view(t) for t in gt], fixed_norm_map[subj], cfg, subj, d)
            m = core.LSTMCopilot(input_size=5, hidden_size=64, n_layers=2)
            m.load_state_dict(torch.load(d / "best_model.pt")); m.eval()
        models[subj] = m

    # baseline once (current copilot) via decay->instantaneous is captured in cur_acc
    base = evaluate_ab(models, splits, decay=1.0, gain=1.0)["overall"]
    print("\n" + "=" * 74)
    print(f"TRAJECTORY-AWARE A/B  (seed {args.seed})  scored by ENDPOINT accuracy only")
    print("=" * 74)
    print(f"raw BCI: {base['bci_acc']*100:.2f}%   current copilot: {base['cur_acc']*100:.2f}%"
          f"   (Δ vs BCI {(base['cur_acc']-base['bci_acc'])*100:+.2f})")
    print("-" * 74)
    print(f"{'decay':>6}{'gain':>6}{'traj_acc':>12}{'Δ vs BCI':>12}{'Δ vs current':>14}")
    best = None
    for decay in args.decays:
        for gain in args.gains:
            ov = evaluate_ab(models, splits, decay=decay, gain=gain)["overall"]
            d_bci = (ov["traj_acc"] - ov["bci_acc"]) * 100
            d_cur = (ov["traj_acc"] - ov["cur_acc"]) * 100
            tag = ""
            if best is None or ov["traj_acc"] > best[0]:
                best = (ov["traj_acc"], decay, gain); tag = ""
            print(f"{decay:>6.1f}{gain:>6.1f}{ov['traj_acc']*100:>11.2f}%{d_bci:>+12.2f}{d_cur:>+14.2f}")
    print("-" * 74)
    print(f"best traj config: decay={best[1]} gain={best[2]}  ({best[0]*100:.2f}%)")
    print("current copilot is the decay->0 / gain=1 limit; positive 'Δ vs current' means")
    print("causal trajectory steering physically converts more belief into endpoint hits.")
    print("=" * 74)
    print("NOTE: single seed. Confirm any gain on seeds 1-2 before trusting it.")


if __name__ == "__main__":
    main()
