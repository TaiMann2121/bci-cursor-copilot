"""
null_control.py  (studies / 8/4 control experiment)
===================================================
The control the +1pp copilot result never had: **does the corrective-velocity
control law still "improve" the endpoint when its classifier carries no target
information at all?**

Motivation (8/4 readout-probe run)
---------------------------------
`experimental/readout_probe.py` showed the copilot endpoint is nearly invariant
to classifier quality: across four training configs the copilot landed at
63.3-64.6% while the classifier driving it swung 46.2-60.4% in aggregate and, on
S04, from 18.5% (fully mode-collapsed onto the most frequent label) to 64.6%. In
one cell S04's classifier sat at 26.7% while the copilot still beat raw by
+4.2pp. A gain that survives its own decoder being broken is not evidence of
decoded intent.

A likely mechanism: `train_copilot.train_one_model` selects its checkpoint on
val *copilot endpoint* accuracy while training on per-tick CE, so a collapsed
classifier can be saved as "best". And `core.corrective_velocity` pushes the
cursor toward a point on the unit target circle, which shrinks/regularizes the
endpoint in a way the direction-only argmax may reward regardless of WHICH
target it aims at.

Design
------
Same held-out test blocks, same control law, same vel_mag. Only the classifier
driving it changes:

  raw       : no copilot (baseline)
  trained   : the real trained copilot                      <- the claimed result
  random    : random-init LSTM, never trained (n seeds)     <- NULL
  shuffled  : trained normally but on SHUFFLED target labels <- NULL
  constant  : always predicts one fixed class                <- degenerate floor

If `random` / `shuffled` / `constant` also beat raw, the reported +1pp is a
property of the control law, not of target inference.

Run:  python studies/null_control.py --seed 0 --clean --sim_frac 0
"""
from __future__ import annotations
import argparse
import contextlib
import dataclasses
import io
from collections import defaultdict

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

import copilot_dataset as cd
import copilot_core as core
from closed_loop import train_copilots


class _ConstRecurrence(nn.Module):
    """Stands in for nn.LSTM: (B,1,F) -> (out, (h, c)) with a dummy 1-wide state."""

    def forward(self, x, hx=None):
        B = x.shape[0]
        z = torch.zeros(B, 1, 1, device=x.device)
        h = torch.zeros(1, B, 1, device=x.device)
        return z, (h, h)


class _ConstHead(nn.Module):
    def __init__(self, k: int, n_classes: int, logit: float):
        super().__init__()
        self.k, self.n_classes, self.logit = k, n_classes, logit

    def forward(self, out):                     # (B,1) -> (B,C)
        o = torch.zeros(out.shape[0], self.n_classes, device=out.device)
        o[:, self.k] = self.logit
        return o


class ConstantModel(nn.Module):
    """Always predicts class k. Exposes the .lstm/.classifier/.input_size surface
    core.simulate_batch steps through, so it drops into the rollout unchanged.

    logit=8.0 makes max-softmax ~0.998, i.e. the control law applies its FULL
    vel_mag every tick. This is deliberately the strongest possible version of
    the degenerate case: a maximally confident, permanently wrong pusher.
    """

    def __init__(self, k: int, n_classes: int = 8, logit: float = 8.0,
                 input_size: int = 5):
        super().__init__()
        self.input_size = input_size        # core.simulate_batch reads this
        self.lstm = _ConstRecurrence()
        self.classifier = _ConstHead(k, n_classes, logit)

    def forward(self, x):                       # (B,T,F) -> (B,T,C)
        B, T, _ = x.shape
        o = torch.zeros(B, T, self.classifier.n_classes, device=x.device)
        o[..., self.classifier.k] = self.classifier.logit
        return o


def rollout_acc(model, test, norm, labels):
    """Endpoint accuracy of the corrective-velocity rollout driven by `model`,
    plus the mean max-softmax confidence over the ticks it actually saw.

    Confidence matters for interpretation: core.simulate_batch scales the
    correction by it (corr = dd * vel_mag * conf), so an untrained near-uniform
    model (conf ~ 1/8) applies ~8x less correction than a confident one. Without
    this column, "random barely moved the endpoint" is uninformative.
    """
    vels = [(t.pos[1:] - t.pos[:-1]).astype(np.float32) for t in test]
    seqs, _, cop = core.simulate_batch(model, vels, norm, "0.02", "basic", "cpu", "additive")
    acc = float((np.array(cop) == labels).mean())

    F = seqs[0].shape[1]
    Tmax = max(s.shape[0] for s in seqs)
    X = np.zeros((len(seqs), Tmax, F), dtype=np.float32)
    M = np.zeros((len(seqs), Tmax), dtype=np.float32)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = s
        M[i, :len(s)] = 1.0
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(X)), dim=-1)
        conf = probs.max(dim=-1).values.numpy()
    mean_conf = float((conf * M).sum() / max(M.sum(), 1e-8))
    return acc, mean_conf


def train_shuffled(splits, seed, sim_frac, rng):
    """Retrain with each subject's target labels permuted (destroys the mapping
    from trajectory to target while preserving label marginals and everything else)."""
    shuffled = {}
    for subj, d in splits.items():
        new = {}
        for part, trajs in d.items():
            if part == "test":
                new[part] = trajs            # test is never touched
                continue
            labs = [t.target_label for t in trajs]
            rng.shuffle(labs)
            # dataclasses.replace keeps session_id/keys intact, so split_real
            # blocking and every downstream provenance check still hold.
            new[part] = [dataclasses.replace(t, target_label=int(l))
                         for t, l in zip(trajs, labs)]
        shuffled[subj] = new
    return train_copilots(shuffled, seed, sim_frac=sim_frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--sim_frac", type=float, default=0.0)
    ap.add_argument("--n_random", type=int, default=3,
                    help="random-init models to average over")
    args = ap.parse_args()

    real_all = cd.load_source("eegk_real", clean=args.clean)
    splits = cd.split_real(real_all, seed=args.seed)
    mix = "real-only" if args.sim_frac <= 0 else f"{1-args.sim_frac:.0%}/{args.sim_frac:.0%} real/sim"
    print(f"Config: seed {args.seed}, {mix}, {'clean' if args.clean else 'raw'} inputs\n")

    print("[1/2] training the real copilots...")
    models, norm_map = train_copilots(splits, args.seed, sim_frac=args.sim_frac)

    print("[2/2] training the shuffled-label copilots...")
    rng = np.random.default_rng(args.seed)
    with contextlib.redirect_stdout(io.StringIO()):
        sh_models, _ = train_shuffled(splits, args.seed, args.sim_frac, rng)

    rows = {}
    tot = defaultdict(float)
    n_tot = 0
    for subj in sorted(models):
        test = splits[subj]["test"]
        if not test:
            continue
        norm = norm_map[subj]
        labels = np.array([t.target_label for t in test])
        n = len(test)

        raw = float((np.array([cd.label_from_position(t.final_pos) for t in test]) == labels).mean())
        trained, c_tr = rollout_acc(models[subj], test, norm, labels)
        shuf, c_sh = rollout_acc(sh_models[subj], test, norm, labels)

        rnd, c_rd = [], []
        for s in range(args.n_random):
            torch.manual_seed(1000 + s)
            m = core.LSTMCopilot(input_size=5, hidden_size=64, n_layers=2).eval()
            a, c = rollout_acc(m, test, norm, labels)
            rnd.append(a); c_rd.append(c)
        rnd_mean, c_rd_mean = float(np.mean(rnd)), float(np.mean(c_rd))

        cs = [rollout_acc(ConstantModel(k), test, norm, labels) for k in range(8)]
        const = float(np.mean([a for a, _ in cs]))
        const_best = float(max(a for a, _ in cs))

        rows[subj] = (n, raw, trained, rnd_mean, shuf, const, const_best,
                      c_tr, c_rd_mean, c_sh)
        for k, v in zip(("raw", "trained", "random", "shuffled", "constant", "const_best",
                         "c_tr", "c_rd", "c_sh"),
                        (raw, trained, rnd_mean, shuf, const, const_best,
                         c_tr, c_rd_mean, c_sh)):
            tot[k] += v * n
        n_tot += n

    print("\n" + "=" * 104)
    print("Endpoint accuracy of the SAME control law under different classifiers "
          "(d = pp vs raw)")
    print(f"{'subj':6}{'N':>6}{'raw':>8}{'trained':>9}{'random':>8}{'shuffled':>10}"
          f"{'const':>8}{'constBest':>10}{'d_train':>8}{'d_rand':>7}{'d_shuf':>7}{'d_const':>8}")
    print("-" * 104)
    for subj, (n, raw, tr, rd, sh, co, cb, *_ ) in rows.items():
        print(f"{subj:6}{n:>6}{raw*100:>7.1f}%{tr*100:>8.1f}%{rd*100:>7.1f}%"
              f"{sh*100:>9.1f}%{co*100:>7.1f}%{cb*100:>9.1f}%"
              f"{(tr-raw)*100:>+8.2f}{(rd-raw)*100:>+7.2f}{(sh-raw)*100:>+7.2f}"
              f"{(co-raw)*100:>+8.2f}")
    print("-" * 104)
    a = {k: tot[k] / n_tot for k in tot}
    print(f"{'ALL':6}{n_tot:>6}{a['raw']*100:>7.1f}%{a['trained']*100:>8.1f}%"
          f"{a['random']*100:>7.1f}%{a['shuffled']*100:>9.1f}%{a['constant']*100:>7.1f}%"
          f"{a['const_best']*100:>9.1f}%"
          f"{(a['trained']-a['raw'])*100:>+8.2f}{(a['random']-a['raw'])*100:>+7.2f}"
          f"{(a['shuffled']-a['raw'])*100:>+7.2f}{(a['constant']-a['raw'])*100:>+8.2f}")
    print("=" * 104)
    print(f"mean correction confidence (scales the nudge):  trained={a['c_tr']:.3f}  "
          f"random={a['c_rd']:.3f}  shuffled={a['c_sh']:.3f}  constant=0.998")
    print("\nrandom    = random-init LSTM, never trained (mean over "
          f"{args.n_random} inits)")
    print("shuffled  = trained on permuted target labels (no trajectory->target mapping)")
    print("const     = always predicts one fixed class, mean over all 8 choices")
    print("constBest = the single best of those 8 fixed choices (an upper bound on")
    print("            what a permanently-wrong confident pusher can score)")
    print("\nREAD: if shuffled/const also beat raw, the reported copilot gain is a")
    print("property of the CONTROL LAW, not of decoded target intent. Compare d_train")
    print("against d_shuf at similar confidence -- that difference, not d_train itself,")
    print("is what target inference is actually worth.")


if __name__ == "__main__":
    main()
