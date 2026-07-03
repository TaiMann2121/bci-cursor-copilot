"""
experiment_anchored.py
======================
Properly-anchored Step-1 comparison. Per subject, splits real into train/val/test
(test = held-out runs, common to all sources). For each source it trains per-subject,
selects the copilot magnitude on val_real, and reports on the pristine test_real.

Sources:
  real      : train on train_real (the sparse baseline the others must beat)
  sim       : train on that subject's EEGK sim (dense, disjoint from real)
  surrogate : train on a dense surrogate CALIBRATED FROM train_real ONLY (no test peek)

Normalization: training features use the training source's norm; val/test simulation
uses norm recomputed from train_real (deployment-realistic real-side calibration).

RUN (one source per call; appends to anchored_results.csv):
    python experiment_anchored.py --source real
    python experiment_anchored.py --source sim
    python experiment_anchored.py --source surrogate
"""
from __future__ import annotations
import argparse, csv, os
import numpy as np, torch
from collections import defaultdict

import copilot_dataset as cd, copilot_core as core
import surrogate_constructor as sc

P1, P2 = 3, 25
EEGK_ROOT = "data/OnlineArmTrajectoryEEGK"
SPLIT_SEED = 0
SURR_SEED = 12345
N_PER_DIR = 200          # dense surrogate


def views_of(trajs):
    return [{"vel": (t.pos[1:] - t.pos[:-1]).astype(np.float32),
             "pos": t.pos.astype(np.float32), "label": t.target_label} for t in trajs]


def train_model(views, norm, vel_mag, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    maxT = max(len(v["vel"]) for v in views)
    m = core.LSTMCopilot(input_size=5, hidden_size=64, n_layers=2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.StepLR(opt, 6, 0.5)
    for ep in range(1, P1 + P2 + 1):
        if ep <= P1:
            seqs = [core.build_sequence_raw(v["vel"], v["pos"], norm, "basic") for v in views]
        else:
            seqs, _, _ = core.simulate_batch(m, [v["vel"] for v in views], norm,
                                             vel_mag, "basic", "cpu", "additive")
        lab = [v["label"] for v in views]
        X = np.zeros((len(seqs), maxT, 5), np.float32); M = np.zeros((len(seqs), maxT), np.float32)
        for i, s in enumerate(seqs):
            k = min(len(s), maxT); X[i, :k] = s[:k]; M[i, :k] = 1
        X = torch.tensor(X); Y = torch.tensor(lab); M = torch.tensor(M)
        perm = torch.randperm(len(X)); m.train()
        for b in range(0, len(X), 128):
            bi = perm[b:b + 128]
            loss = core.masked_weighted_ce(m(X[bi]), Y[bi], M[bi], "exponential", 3.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
    return m


def acc_on(model, trajs, norm, vel_mag):
    vels = [(t.pos[1:] - t.pos[:-1]).astype(np.float32) for t in trajs]
    _, _, preds = core.simulate_batch(model, vels, norm, vel_mag, "basic", "cpu", "additive")
    labs = np.array([t.target_label for t in trajs])
    return float((preds == labs).mean())


def build_surrogate(cal_trajs, subj, stats):
    prof = sc.measure_profile(cal_trajs)
    rng = np.random.default_rng(SURR_SEED)
    sampler = sc.make_endpoint_sampler(prof, subj, stats, "geometric", rng)
    df = sc.generate_subject(subj, prof, sampler, N_PER_DIR, 1.0, rng)
    return cd._dataframe_to_trajectories(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["real", "sim", "surrogate"])
    ap.add_argument("--mags", default="0.005,0.01,0.02,0.035")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--split_seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--subjects", default="", help="comma-sep subset, e.g. S01,S02")
    ap.add_argument("--out", default="anchored_results.csv")
    args = ap.parse_args()
    mags = [float(x) for x in args.mags.split(",")]
    seeds = list(range(args.seeds))

    # resume: skip (split_seed,source,subject,seed,vel_mag) already present
    done = set()
    if os.path.exists(args.out):
        import csv as _csv
        with open(args.out) as _f:
            for r in _csv.DictReader(_f):
                done.add((int(r.get("split_seed", 0)), r["source"], r["subject"],
                          int(r["seed"]), float(r["vel_mag"])))

    real = [t for t in cd.load_source("eegk_real", repo_root=".") if int(t.keys[2]) != 1]
    split = cd.split_real(real, seed=args.split_seed)
    subjects = sorted(split)
    if args.subjects:
        want = set(args.subjects.split(","))
        subjects = [s for s in subjects if s in want]
    if args.subjects:
        want = set(args.subjects.split(","))
        subjects = [s for s in subjects if s in want]

    sim_by = defaultdict(list)
    if args.source == "sim":
        for t in cd.load_source("eegk_sim", repo_root="."):
            sim_by[t.subject_id].append(t)
    stats = cd.load_typing_stats(EEGK_ROOT) if args.source == "surrogate" else {}

    new = not os.path.exists(args.out)
    f = open(args.out, "a", newline=""); w = csv.writer(f)
    if new:
        w.writerow(["split_seed", "source", "subject", "seed", "vel_mag", "n_train",
                    "val_acc", "test_acc", "test_bci"])

    for s in subjects:
        if all((args.split_seed, args.source, s, seed, mag) in done for seed in seeds for mag in mags):
            continue
        tr, va, te = split[s]["train"], split[s]["val"], split[s]["test"]
        eval_norm = cd.compute_norm_stats(tr)                 # real-side norm
        test_bci = float(np.mean([cd.label_from_position(t.final_pos) == t.target_label for t in te]))

        if args.source == "real":
            train_trajs = tr
        elif args.source == "sim":
            train_trajs = sim_by[s]
        else:
            train_trajs = build_surrogate(tr, s, stats)
        tv = views_of(train_trajs)
        train_norm = cd.compute_norm_stats(train_trajs)

        for seed in seeds:
            for mag in mags:
                if (args.split_seed, args.source, s, seed, mag) in done:
                    continue
                model = train_model(tv, train_norm, mag, seed)
                va_acc = acc_on(model, va, eval_norm, mag)
                te_acc = acc_on(model, te, eval_norm, mag)
                w.writerow([args.split_seed, args.source, s, seed, mag, len(train_trajs),
                            f"{va_acc:.4f}", f"{te_acc:.4f}", f"{test_bci:.4f}"]); f.flush()
        print(f"  {args.source} {s}: n_train={len(train_trajs)} done "
              f"(bci_test={test_bci*100:.1f}%)")
    f.close()
    print(f"[{args.source}] appended to {args.out}")


if __name__ == "__main__":
    main()
